import torch
import torch.nn.functional as F
import math
import torch.nn as nn

from lib.xttn import mask_xattn_one_text


def is_sqr(n):
    a = int(math.sqrt(n))
    return a * a == n


class TokenSparse(nn.Module):
    def __init__(self, embed_dim=512, sparse_ratio=0.6):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio
    
    def forward(self, tokens, attention_x, attention_y):
        
        B_v, L_v, C = tokens.size()

        # (B_v, L_v)
        score = attention_x + attention_y

        num_keep_token = math.ceil(L_v * self.sparse_ratio)
    
        # select the top-k index, (B_v, L_v)
        score_sort, score_index = torch.sort(score, dim=1, descending=True)
        
        # (B_v, L_v * token_ratio)
        keep_policy = score_index[:, :num_keep_token]

        # (B_v, L_v)
        score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1)
        
        # (B_v, L_v * token_ratio, C)
        select_tokens = torch.gather(tokens, dim=1, index=keep_policy.unsqueeze(-1).expand(-1, -1, C))

        # fusion token
        # (B_v, L_v *  (1 - token_ratio) )
        non_keep_policy = score_index[:, num_keep_token:]

        # (B_v, L_v *  (1 - token_ratio), C )
        non_tokens = torch.gather(tokens, dim=1, index=non_keep_policy.unsqueeze(-1).expand(-1, -1, C))
        
        # (B_v, L_v *  (1 - token_ratio) )
        non_keep_score = score_sort[:, num_keep_token:]
        # through softmax function, (B_v, L_v *  (1 - token_ratio) ) -> (B_v, L_v *  (1 - token_ratio), 1)
        non_keep_score = F.softmax(non_keep_score, dim=1).unsqueeze(-1)

        # get fusion token (B_v, 1, C)
        extra_token = torch.sum(non_tokens * non_keep_score, dim=1, keepdim=True) 

        return select_tokens, extra_token, score_mask
                  

# dim_ratio affect GPU memory
class TokenAggregation(nn.Module):
    def __init__(self, dim=512, keeped_patches=64, dim_ratio=0.2):
        super().__init__()
        
        hidden_dim = int(dim * dim_ratio)

        self.weight = nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, keeped_patches)
                        )
        
        self.scale = nn.Parameter(torch.ones(1, 1, 1))
        
    def forward(self, x, keep_policy=None):

        # (B, N, C) -> (B, N, N_s)
        weight = self.weight(x)

        #  (B, N, N_s) -> (B, N_s, N)
        weight = weight.transpose(2, 1) * self.scale       

        if keep_policy is not None:
            # keep_policy (B, N) -> (B, 1, N)
            keep_policy = keep_policy.unsqueeze(1)
            # increase a large number for mask patches
            weight = weight - (1 - keep_policy) * 1e10

        # learning a set of weight matrices
        weight = F.softmax(weight, dim=2)
        
        # (B, N_s, C)
        # multiply with patch features
        x = torch.bmm(weight, x)
        
        return x
    

## sparse + aggregation
class CrossSparseAggrNet_v2(nn.Module):
    def __init__(self, opt=None):
        super().__init__()

        self.opt = opt
        
        self.hidden_dim = opt.hidden_size  
        self.num_patches = 196

        self.sparse_ratio = opt.sparse_ratio 
        self.aggr_ratio = opt.aggr_ratio 

        self.attention_weight = opt.attention_weight
        self.ratio_weight = opt.ratio_weight
        
        # the number of aggregated patches
        self.keeped_patches = int(self.num_patches * self.aggr_ratio * self.sparse_ratio)

        # sparse network
        self.sparse_net_cap = TokenSparse(embed_dim=self.hidden_dim, 
                                      sparse_ratio=self.sparse_ratio,
                                      )
        self.sparse_net_long = TokenSparse(embed_dim=self.hidden_dim, 
                                      sparse_ratio=self.sparse_ratio,
                                    )
        # aggregation network
        self.aggr_net= TokenAggregation(dim=self.hidden_dim, 
                                        keeped_patches=self.keeped_patches,
                                        )  

    def forward(self, img_embs, cap_embs, cap_lens, long_cap_embs=None, long_cap_lens=None, epoch=None, max_epoch=None, return_debug_masks=False):

        B_v, L_v, C = img_embs.shape
    
        # feature normalization
        # (B_v, L_v, C)
        img_embs_norm = F.normalize(img_embs, dim=-1)
        # (B_t, L_t, C)
        cap_embs_norm = F.normalize(cap_embs, dim=-1)


        long_cap_embs_norm = F.normalize(long_cap_embs, dim=-1)

        self.has_cls_token = False if is_sqr(img_embs.shape[1]) else True

        #  whether it exists [cls] token
        if self.has_cls_token:
            # (B_v, 1, C)
            img_cls_emb = img_embs[:, 0:1, :]
            img_cls_emb_norm = img_embs_norm[:, 0:1, :]
            img_spatial_embs = img_embs[:, 1:, :]
            img_spatial_embs_norm = img_embs_norm[:, 1:, :]
        else:
            img_spatial_embs = img_embs
            img_spatial_embs_norm = img_embs_norm



        # compute self-attention 
        with torch.no_grad():
            # (B_v, L_v, C) ->  (B_v, 1, C)
            img_spatial_glo_norm = F.normalize(img_spatial_embs.mean(dim=1, keepdim=True), dim=-1)
            # (B_v, L_v, C) -> (B_v, L_v)
            img_spatial_self_attention = (img_spatial_glo_norm * img_spatial_embs_norm).sum(dim=-1)

       
       
       
        improve_sims = []
        long_sims= []
        score_mask_all = []
        score_mask_long_all = []
        selected_tokens_all = []
        selected_tokens_long_all = []

        # Introduce text information
        # process each text separately
        for i in range(len(cap_lens)):
            # Ensure that n_word does not exceed the actual number of words in cap_embs[i]
            n_word = int(cap_lens[i])

            if i >= cap_embs.size(0):
                # Skip iteration if the index exceeds available elements in cap_embs
                continue

            # Clamp n_word to the maximum possible length in cap_embs[i]
            n_word = min(n_word, cap_embs.size(1))

            # (L_t, C)
            cap_i = cap_embs[i, :n_word, :]

            # (B_v, L_t, C)
            cap_i_expand = cap_embs_norm[i, :n_word, :].unsqueeze(0).repeat(B_v, 1, 1)

            ## compute cross-attention
            with torch.no_grad():
                # (L_t, C) -> (1, C) -> (1, 1, C)
                cap_i_glo = F.normalize(cap_i.mean(0, keepdim=True).unsqueeze(0), dim=-1)
                # (B_v, L_v, C) -> (B_v, L_v)
                # img_spatial_cap_i_attention = (cap_i_glo * img_spatial_embs_norm).sum(dim=-1)
                attn_cap = (cap_i_glo * img_spatial_embs_norm).sum(dim=-1)
                select_tokens_cap, extra_token_cap, score_mask_cap = self.sparse_net_cap(
                tokens=img_spatial_embs,
                attention_x=img_spatial_self_attention,
                attention_y=attn_cap,
                )
            # selection
            #select_tokens, extra_token, score_mask = self.sparse_net(tokens=img_spatial_embs, 
            #                                                        attention_x=img_spatial_self_attention, 
            #                                                        attention_y=img_spatial_cap_i_attention,
            #                                                       )

            # aggregation
            aggr_tokens = self.aggr_net(select_tokens_cap)
            # aggr_tokens = select_tokens

            # add fusion token
            keep_spatial_tokens = torch.cat([aggr_tokens, extra_token_cap], dim=1)

            # add [cls] token
            if self.has_cls_token:
                select_tokens = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)
            else:
                select_tokens = keep_spatial_tokens

            # patch normalization
            select_tokens = F.normalize(select_tokens, dim=-1)

            # image-text similarity
            # (B_v, 1)
            sim_one_text = mask_xattn_one_text(img_embs=select_tokens, 
                                            cap_i_expand=cap_i_expand,
                                            )

            improve_sims.append(sim_one_text)
            score_mask_all.append(score_mask_cap)
            selected_tokens_all.append(select_tokens)




        for i in range(len(long_cap_lens)):
            n_word = int(long_cap_lens[i])

            if i >= long_cap_embs.size(0):
                continue

            n_word = min(n_word, long_cap_embs.size(1))

            long_cap_i = long_cap_embs[i, :n_word, :]
            long_cap_i_expand = long_cap_embs_norm[i, :n_word, :].unsqueeze(0).repeat(B_v, 1, 1)

            # cap_emb cross-attention & sparse selection
            with torch.no_grad():   
                long_cap_i_glo = F.normalize(long_cap_i.mean(0, keepdim=True).unsqueeze(0), dim=-1)
                long_attn_cap = (long_cap_i_glo * img_spatial_embs_norm).sum(dim=-1)
                select_tokens_long, extra_token_long, score_mask_long = self.sparse_net_long(
                tokens=img_spatial_embs,
                attention_x=img_spatial_self_attention,
                attention_y=long_attn_cap,
                )

            # aggregation
            aggr_tokens_long = self.aggr_net(select_tokens_long)

            # add fusion token
            keep_spatial_tokens = torch.cat([aggr_tokens_long, extra_token_long], dim=1)

            # add [cls] token
            if self.has_cls_token:
                select_tokens_long = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)
            else:
                select_tokens_long = keep_spatial_tokens

            # patch normalization
            select_tokens_long = F.normalize(select_tokens_long, dim=-1)


            # image-text similarity
            sim_one_text = mask_xattn_one_text(
                img_embs=select_tokens_long,
                cap_i_expand=long_cap_i_expand,
            )

            long_sims.append(sim_one_text)
            score_mask_long_all.append(score_mask_long)
            selected_tokens_long_all.append(select_tokens_long)

        # (B_v, B_t)
        improve_sims = torch.cat(improve_sims, dim=1)
        if len(long_sims) > 0:
            improve_sims = improve_sims + torch.cat(long_sims, dim=1)

        cap_mask_all = torch.stack(score_mask_all, dim=0)
        long_mask_all = torch.stack(score_mask_long_all, dim=0) if len(score_mask_long_all) > 0 else None

        score_mask_all = cap_mask_all
        if long_mask_all is not None:
            # Keep score mask in [0, 1] for sparse-ratio regularization.
            score_mask_all = 0.5 * (score_mask_all + long_mask_all)

        def _matched_tokens(token_list):
            token_stack = torch.stack(token_list, dim=0)
            if token_stack.size(0) == B_v:
                idx = torch.arange(B_v, device=token_stack.device)
                return token_stack[idx, idx]
            return token_stack.mean(dim=0)

        image_tokens = _matched_tokens(selected_tokens_all)
        if len(selected_tokens_long_all) > 0:
            image_tokens_long = _matched_tokens(selected_tokens_long_all)
            image_tokens = 0.5 * (image_tokens + image_tokens_long)

        image_tokens = F.normalize(image_tokens, dim=-1)

        if return_debug_masks:
            return improve_sims, score_mask_all, image_tokens, cap_mask_all, long_mask_all

        return improve_sims, score_mask_all, image_tokens
        # if self.training:
        #     return improve_sims, score_mask_all
        # else:
        #     return improve_sims


if __name__ == '__main__':

    pass