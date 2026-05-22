import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, GenerationConfig, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from lib.cross_net import CrossSparseAggrNet_v2
from torch.nn.utils.rnn import pad_sequence
from lib.loss import loss_select

from attention import DotProductAttention

from torch_geometric.nn import GCNConv
import sys
# import spacy
import math
from torch.nn import CrossEntropyLoss

class DynamicGraphConstructor(nn.Module):
    def __init__(self, hidden_dim=768, k_neighbors=5):
        super().__init__()
        self.k = k_neighbors
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        edge_indices = []
        for b in range(batch_size):
            
            sim_matrix = torch.matmul(x[b], x[b].T)  # [seq_len, seq_len]
            
            _, topk_indices = torch.topk(sim_matrix, self.k, dim=1)
            
            src = torch.repeat_interleave(torch.arange(seq_len, device=x.device), self.k)
            dst = topk_indices.view(-1)
            
            edge_index = torch.stack([torch.cat([src, dst]),
                                      torch.cat([dst, src])], dim=0)
            
            edge_index = torch.unique(edge_index, dim=1)
            edge_indices.append(edge_index)
        return edge_indices


class GCNTransformerLayer(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8):
        super().__init__()
        
        self.gcn_conv = GCNConv(hidden_dim, hidden_dim)

        
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        
        batch_size = x.size(1)
        x_gcn = x.permute(1, 0, 2)  # [batch_size, seq_len, hidden_dim]

        
        gcn_out = []
        for b in range(batch_size):
            gcn_out.append(self.gcn_conv(x_gcn[b], edge_index[b]))
        gcn_out = torch.stack(gcn_out, dim=1)  # [seq_len, batch_size, hidden_dim]

        
        attn_out, _ = self.self_attn(x, x, x)  # [seq_len, batch_size, hidden_dim]

        
        combined = torch.cat([gcn_out, attn_out], dim=-1)  # [seq_len, batch_size, 2*hidden_dim]
        gate = self.fusion_gate(combined)  # [seq_len, batch_size, hidden_dim]
        fused = gate * gcn_out + (1 - gate) * attn_out
        
        output = self.norm2(fused + x)
        return output


class GCNTransformer(nn.Module):
    def __init__(self, num_layers=3):
        super().__init__()

        self.graph_builder = DynamicGraphConstructor()
        self.layers = nn.ModuleList([
            GCNTransformerLayer() for _ in range(num_layers)
        ])

        
        self.output_proj = nn.Linear(768, 768)

    def forward(self, x):

        
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]

        
        edge_indices = self.graph_builder(x.permute(1, 0, 2))  # 恢复batch_first

        
        for layer in self.layers:
            x = layer(x, edge_indices)

        
        output = x.permute(1, 0, 2)  # [batch_size, seq_len, hidden_dim]

        
        output = self.output_proj(output)

        return output



class T5TextSentimentEnhancer(nn.Module):
    """
    Auxiliary Text Sentiment Enhancement Module for T5
    Operates on T5 encoder hidden states only.
    """
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        gcn_layers: int = 3,
        dropout_prob: float = 0.1
    ):
        super().__init__()

        # GCN + Transformer structure enhancement
        self.gcn_transformer = GCNTransformer(num_layers=gcn_layers)

        # Token-level classifier
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Loss
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        encoder_hidden_states,   # [B, L, H]
        attention_mask=None,     # [B, L]
        labels=None              # [B, L] or None
    ):
        """
        Returns:
            dict:
                enhanced_states
                logits
                loss (optional)
        """

        # 1. GCN-based structure enhancement
        enhanced_states = self.gcn_transformer(encoder_hidden_states)

        # 2. Token-level prediction
        enhanced_states = self.dropout(enhanced_states)
        logits = self.classifier(enhanced_states)

        outputs = {
            "enhanced_states": enhanced_states,
            "logits": logits
        }

        # 3. Auxiliary sentiment loss
        if labels is not None:
            if labels.dim() == 1:
                if attention_mask is None:
                    raise ValueError("attention_mask is required for sentence-level sentiment loss")
                mask = attention_mask.unsqueeze(-1).type_as(enhanced_states)
                pooled = (enhanced_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                sent_logits = self.classifier(self.dropout(pooled))
                outputs["sent_logits"] = sent_logits
                outputs["loss"] = self.loss_fct(sent_logits, labels)
            else:
                loss = self.loss_fct(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                outputs["loss"] = loss

        return outputs
        
class CARSA(nn.Module):
    def __init__(self, args) -> None:
        super(CARSA, self).__init__()
        self.args = args

        self.t5 = T5ForConditionalGeneration.from_pretrained(args.pretrained_model_dir)
        self.t5.resize_token_embeddings(len(args.tokenizer))
        self.text_embeddings = self.t5.get_input_embeddings()
        

        self.img_fc = nn.Linear(args.img_hidden_size, args.hidden_size)
        
        
        
        self.dropout_prob = 0.1
        self.dropout = nn.Dropout(self.dropout_prob)
        self.att = DotProductAttention(dropout=self.dropout_prob)

        self.a_generation_config = GenerationConfig.from_pretrained(args.pretrained_model_config_dir, 'a_generation_config.json')
        self.ea_generation_config = GenerationConfig.from_pretrained(args.pretrained_model_config_dir, 'ea_generation_config.json')
        self.iea_generation_config = GenerationConfig.from_pretrained(args.pretrained_model_config_dir, 'iea_generation_config.json')

        self.cross_net = CrossSparseAggrNet_v2(args)
        self.criterion = loss_select(args, loss_type=args.loss)

        self.beta = args.beta
        self.gamma = args.gamma
        self.sparse_ratio = args.sparse_ratio
        self.ratio_weight = args.ratio_weight

        self.text_sentiment_enhancer = T5TextSentimentEnhancer(
            hidden_size=args.hidden_size,
            num_labels=3,
            gcn_layers=3,
            dropout_prob=0.1
        )

    def forward(self, a_input_ids, a_attention_mask, a_decoder_output_labels, ea_input_ids, ea_attention_mask, ea_decoder_output_labels, iea_input_ids,
                iea_attention_mask, iea_decoder_output_labels, image_feature, cap_input_ids, cap_attention_mask, imgid, long_cap_input_ids, long_cap_attention_mask,
                caption_input_ids=None, caption_attention_mask=None,
                is_eval=False, a_sentiment_labels=None, epoch=None, max_epoch=None, sentiment_token_labels=None):
        img_feat = self.img_fc(image_feature)
        cap_encoder_inputs_embeds = self.text_embeddings(cap_input_ids)
        cap_lens = cap_attention_mask.sum(dim=1).long()

        long_cap_encoder_inputs_embeds = self.text_embeddings(long_cap_input_ids)
        long_cap_lens = long_cap_attention_mask.sum(dim=1).long()

        improved_sims, score_mask_all, image_tokens = self.cross_net(
            img_feat,
            cap_encoder_inputs_embeds,
            cap_lens,
            long_cap_encoder_inputs_embeds,
            long_cap_lens,
            epoch=epoch,
            max_epoch=max_epoch,
        )
       

        
        # ===== 1. T5 Encoder =====
        encoder_outputs = self.t5.encoder(
            input_ids=a_input_ids,
            attention_mask=a_attention_mask,
            return_dict=True
        )

        encoder_hidden_states = encoder_outputs.last_hidden_state
        # [B, L, H]

        # ===== 2. Auxiliary sentiment enhancement =====
        sentiment_outputs = self.text_sentiment_enhancer(
            encoder_hidden_states,
            attention_mask=a_attention_mask,
            labels=sentiment_token_labels   # sentence-level or token-level sentiment labels
        )

        text_loss = sentiment_outputs.get("loss")
        if text_loss is None:
            text_loss = torch.tensor(0.0, device=encoder_hidden_states.device)

        

        a_encoder_inputs_embeds = self.text_embeddings(a_input_ids)  # (B, L, H)

        ea_encoder_inputs_embeds = self.text_embeddings(ea_input_ids)  # (B, L, H)


        iea_encoder_inputs_embeds = self.text_embeddings(iea_input_ids)  # (B, L, H)

        caption_encoder_inputs_embeds = self.text_embeddings(caption_input_ids)

        
        
        a_encoder_att_output, _ = self.att(cap_encoder_inputs_embeds, a_encoder_inputs_embeds, a_encoder_inputs_embeds, a_attention_mask.sum(dim=-1))
        ea_encoder_att_output, _ = self.att(cap_encoder_inputs_embeds, ea_encoder_inputs_embeds, ea_encoder_inputs_embeds, ea_attention_mask.sum(dim=-1))
        iea_encoder_att_output, _ = self.att(cap_encoder_inputs_embeds, iea_encoder_inputs_embeds, iea_encoder_inputs_embeds, iea_attention_mask.sum(dim=-1))

        a_cap_encoder_inputs_embeds = (a_encoder_att_output + cap_encoder_inputs_embeds) / 2
        ea_cap_encoder_inputs_embeds = (ea_encoder_att_output + cap_encoder_inputs_embeds) / 2
        iea_cap_encoder_inputs_embeds = (iea_encoder_att_output + cap_encoder_inputs_embeds) / 2

        a_encoder_att_output, _ = self.att(a_cap_encoder_inputs_embeds, caption_encoder_inputs_embeds, caption_encoder_inputs_embeds, caption_attention_mask.sum(dim=-1))
        ea_encoder_att_output, _ = self.att(ea_cap_encoder_inputs_embeds, caption_encoder_inputs_embeds, caption_encoder_inputs_embeds, caption_attention_mask.sum(dim=-1))
        iea_encoder_att_output, _ = self.att(iea_cap_encoder_inputs_embeds, caption_encoder_inputs_embeds, caption_encoder_inputs_embeds, caption_attention_mask.sum(dim=-1))
        




        a_encoder_inputs_embeds = torch.cat([a_encoder_inputs_embeds, a_encoder_att_output, caption_encoder_inputs_embeds], dim=1)
        ea_encoder_inputs_embeds = torch.cat([ea_encoder_inputs_embeds, ea_encoder_att_output, caption_encoder_inputs_embeds], dim=1)
        iea_encoder_inputs_embeds = torch.cat([iea_encoder_inputs_embeds, iea_encoder_att_output, caption_encoder_inputs_embeds], dim=1)

        a_attention_mask = torch.cat([a_attention_mask, cap_attention_mask, caption_attention_mask], dim=1)
        ea_attention_mask = torch.cat([ea_attention_mask, cap_attention_mask, caption_attention_mask], dim=1)
        iea_attention_mask = torch.cat([iea_attention_mask, cap_attention_mask, caption_attention_mask], dim=1)


        image_attention_mask = torch.ones(
            image_tokens.size(0),
            image_tokens.size(1),
            dtype=a_attention_mask.dtype,
            device=a_attention_mask.device,
        )

        a_encoder_inputs_embeds = torch.cat([a_encoder_inputs_embeds, image_tokens], dim=1)
        ea_encoder_inputs_embeds = torch.cat([ea_encoder_inputs_embeds, image_tokens], dim=1)
        iea_encoder_inputs_embeds = torch.cat([iea_encoder_inputs_embeds, image_tokens], dim=1)

        a_attention_mask = torch.cat([a_attention_mask, image_attention_mask], dim=1)
        ea_attention_mask = torch.cat([ea_attention_mask, image_attention_mask], dim=1)
        iea_attention_mask = torch.cat([iea_attention_mask, image_attention_mask], dim=1)

       



        if not is_eval: 
            #a_t5_output = self.t5(encoder_outputs=a_encoder_outputs, attention_mask=a_attention_mask, labels=a_decoder_output_labels)
            a_t5_output = self.t5(inputs_embeds=a_encoder_inputs_embeds, attention_mask=a_attention_mask, labels=a_decoder_output_labels)
            a_loss = a_t5_output.loss

            ea_t5_output = self.t5(inputs_embeds=ea_encoder_inputs_embeds, attention_mask=ea_attention_mask, labels=ea_decoder_output_labels)
            ea_loss = ea_t5_output.loss

            iea_t5_output = self.t5(inputs_embeds=iea_encoder_inputs_embeds, attention_mask=iea_attention_mask, labels=iea_decoder_output_labels)
            iea_loss = iea_t5_output.loss

            align_loss = self.criterion(img_feat, cap_encoder_inputs_embeds, imgid, improved_sims)
            ratio_loss = (score_mask_all.mean() - self.sparse_ratio) ** 2
            loss = align_loss + self.ratio_weight * ratio_loss
            a_loss = a_loss + loss * self.beta + self.gamma * text_loss
            return a_loss, ea_loss, iea_loss

        else:
            a_sequence_ids = self.t5.generate(inputs_embeds=a_encoder_inputs_embeds, attention_mask=a_attention_mask, generation_config=self.a_generation_config)
            ea_sequence_ids = self.t5.generate(inputs_embeds=ea_encoder_inputs_embeds, attention_mask=ea_attention_mask, generation_config=self.ea_generation_config)
            iea_sequence_ids = self.t5.generate(inputs_embeds=iea_encoder_inputs_embeds, attention_mask=iea_attention_mask, generation_config=self.iea_generation_config)
            a_sequence = self.args.tokenizer.batch_decode(a_sequence_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            ea_sequence = self.args.tokenizer.batch_decode(ea_sequence_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            iea_sequence = self.args.tokenizer.batch_decode(iea_sequence_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            return a_sequence, ea_sequence, iea_sequence


