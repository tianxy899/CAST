# Introduction

This repository contains the implementation for the paper "Improving End-to-end Speech-to-text Translation with Document-level Context".

## Environment Configuration

1. Clone this repository:

```
git clone git@github.com:tianxy899/CAST.git
cd CAST/
```

2. Install `fairseq`:

```
pip install --editable ./
python setup.py build develop
```

3. We organize our implementation as fairseq plug-ins in the  `cast` directory:

```
.
├── __init__.py
├── criterions
│   ├── __init__.py
│   ├── speech_and_text_translation_criterion.py
│   └── speech_and_text_translation_criterion_mix_jsd_gate.py
├── datasets
│   ├── __init__.py
│   ├── audio_utils.py
│   ├── speech_and_text_translation_dataset.py
│   ├── speech_and_text_translation_dataset_d2s_gate.py
│   ├── speech_and_text_translation_dataset_d2s_gate_infer.py
│   └── speech_to_text_dataset.py
├── models
│   ├── __init__.py
│   ├── hubert_transformer.py
│   └── hubert_transformer_d2s_gate.py
├── prepare_data
│   ├── README.md
│   ├── apply_spm.py
│   ├── data_utils.py
│   └── prep_mustc_data.py
├── scripts
│   ├── average_checkpoints.py
│   ├── infer_d2s.sh
│   ├── infer_sent.sh
│   ├── prepare_extra_mt.sh
│   ├── pretrain_mt.sh
│   ├── train_en2x_extmt.sh
│   └── train_en2x_extmt_d2s.py
└── tasks
    ├── __init__.py
    ├── speech_and_text_translation.py
    └── speech_to_text_modified.py
```

You can import our implementation with `--user-dir cast` in fairseq.



## Data Preparation

1. External MT data: Download the external MT data (see our paper for details) and run the script:

```
sh cast/scripts/prepare_extra_mt.sh
```

2. MuST-C data: Download the [MuST-C v1.0](https://ict.fbk.eu/must-c/) and run the script:

```
python cast/prepare_data/prep_mustc_data.py
```


## Model Training

The modal training contains three steps: MT pre-training, sentence-level ST pre-
training and document-level context-aware ST fine-tuning.

All the training scripts below are configured to run using **4 GPUs**. You can adjust `--update-freq` depending on the number of your available GPUs.

Before training, please download the [HuBERT-Base](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt) model and place it in the `checkpoints/hubert_base_ls960.pt` path.

### MT Pre-training (Optional) 

Pre-train the model with the external MT dataset. Please run the script:

```
sh cast/scripts/pretrain_mt.sh
```

You should adjust the maximum training steps (`--max-update`) based on the size of the training data.

After training, please average the last 5 checkpoints:

```
python cast/scripts/average_checkpoints.py \
    --inputs checkpoints/en-de/pretrain_mt \
    --num-epoch-checkpoints 5 \
    --output checkpoints/en-de/pretrain_mt/avg_last_5_epoch.pt
```

### Sentence-level ST Pre-training

For the sentence-level pre-training, please run the script:

```
sh cast/scripts/train_en2x_extmt.sh
```
After training, please average the last 10 checkpoints after selecting the best checkpoint on the `dev` set.

### Document-level Context-aware ST Fine-tuning (CAST)

For the `CAST` training, please run the script:

```
sh cast/scripts/train_en2x_extmt_d2s.sh
```
After training, please average the last 10 checkpoints after selecting the best checkpoint on the `dev` set.


## Evaluation

### Sentence-level Evaluation

To evaluation the performance of the sentence-level ST model, please use the `cast/scripts/infer_sent.sh` script:

```
sh cast/scripts/infer_sent.sh
```

### Document-level Evaluation

To evaluation the performance of the document-level ST model, please use the `cast/scripts/infer_d2s.sh` script.

```
sh cast/scripts/infer_d2s.sh
```


## Citation

If this repository is useful for you, please cite as:

```
@ARTICLE{tian-2025-improving,
      title={Improving End-to-End Speech-to-Text Translation With Document-Level Context}, 
      author={Tian, Xinyu and Wei, Haoran and Gong, Zhengxian and Li, Junhui and Xie, Jun},
      journal={IEEE Transactions on Audio, Speech and Language Processing}, 
      year={2025},
      volume={33},
      pages={2098-2109},
}
```
