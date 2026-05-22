# Environment Requirements

We recommend using conda for environment management:

```bash
conda env create -f CARSA.yaml
```

# Datasets
1.Constructed datasets: Twitter2015 ([twitter2015](./data/twitter2015)), Twitter2017 ([twitter2015](./data/twitter2017)).

2.Image features can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1RiFUQpOMSX9mjJYypylxyul_2Pw7v8Fn?usp=drive_link). Place the downloaded files into the `data/twitter2015` and `data/twitter2017` directories respectively, and name them `images_feature`.

# Pretrained Model
The Flan-T5 model is utilized as the backbone. Download the pre-trained model [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) and save it in the directory `pretrained/flan-t5-base`.

# Training & Evaluation
To train and evaluate the CARSA model on different datasets, run the following commands:

Run on Twitter2015 dataset
```bash
python run_carsa_15.py
```
Run on Twitter2017 dataset
```bash
python run_carsa_17.py
```
