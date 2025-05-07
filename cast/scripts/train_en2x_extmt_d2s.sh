#!/usr/bin/env bash

TGT_LANG=de
# MODEL_DIR=./checkpoints/en-$TGT_LANG/st_extmt_doc_gate_loss_lambda0.3
MODEL_DIR=./checkpoints/en-$TGT_LANG/test1
HUBERT=./checkpoints/hubert_base_ls960.pt
FAIRSEQ=./fairseq_cli
DATA=./data/en-$TGT_LANG/doc-data-hubert
PRETRAIN=/public/home/zhxgong/xytian/fairseq/checkpoints/en-de/st_baseline_extmt/st_baseline_extmt_27.83.pt
TEXTDATA=/public/home/zhxgong/xytian/fairseq/data/en-$TGT_LANG/extra_mt/bin     # no use

CUDA_VISIBLE_DEVICES=0,1,2,3 python $FAIRSEQ/train.py $DATA \
    --user-dir cress \
    --st-training --mt-finetune \
    --text-data $TEXTDATA \
    --task speech_and_text_translation --tgt-lang $TGT_LANG \
    --train-subset train --valid-subset dev \
    --config-yaml config.yaml --num-workers 4 \
    --max-audio-positions 1400000 --max-source-positions 1024 --max-target-positions 1024 \
    --max-tokens 2000000 --max-text-tokens 2000 --max-tokens-valid 2000000 \
    --skip-invalid-size-inputs-valid-test \
    \
    --arch hubert_transformer_d2s_gate_postln --hubert-model-path $HUBERT \
    --layernorm-embedding \
    \
    --optimizer adam --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --lr 1e-4 --warmup-updates 4000 --weight-decay 0.0001 \
    \
    --criterion speech_and_text_translation_mix_jsd_gate --doc-mode-gate --mix-ratio 0.4 --jsd-weight 1.0 --gate-weight 0.3 \
    --label-smoothing 0.1 --report-accuracy \
    \
    --update-freq 2 --max-update 500000 \
    \
    --no-progress-bar --log-format json --log-interval 10 \
    --save-interval-updates 1000 --no-epoch-checkpoints \
    --save-dir ${MODEL_DIR} \
    --distributed-world-size 4 --ddp-backend=no_c10d --fp16 \
    \
    --eval-bleu --eval-bleu-args '{"beam": 8}' \
    --eval-bleu-detok moses --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    \
    1>./log/en-$TGT_LANG/xytian1.log 2>&1