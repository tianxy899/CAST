#!/usr/bin/env bash

TGT_LANG=de
MODEL_DIR=checkpoints/en-$TGT_LANG/sentence_level_st
HUBERT=checkpoints/hubert_base_ls960.pt
DATA=data/en-$TGT_LANG
PRETRAIN=checkpoints/en-$TGT_LANG/pretrain_mt/avg_last_5_epoch.pt

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train $DATA \
    --user-dir cast \
    --st-training --mt-finetune \
    --task speech_and_text_translation --tgt-lang $TGT_LANG \
    --train-subset train --valid-subset dev \
    --config-yaml config.yaml --num-workers 4 \
    --max-tokens 2000000 --batch-size 32 --max-text-tokens 4096 --max-tokens-valid 2000000 \
    --skip-invalid-size-inputs-valid-test \
    \
    --arch hubert_transformer_postln --hubert-model-path $HUBERT \
    --layernorm-embedding \
    \
    --optimizer adam --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --lr 1e-4  --warmup-updates 4000  --weight-decay 0.0001 \
    \
    --criterion speech_and_text_translation \
    --label-smoothing 0.1 --report-accuracy \
    \
    --update-freq 2 --max-update 200000 \
    \
    --no-progress-bar --log-format json --log-interval 100 \
    --save-interval-updates 1000 --no-epoch-checkpoints \
    --save-dir ${MODEL_DIR} --patience 10 \
    --distributed-world-size 4 --ddp-backend=no_c10d --fp16 \
    \
    --eval-bleu --eval-bleu-args '{"beam": 8}' \
    --eval-bleu-detok moses --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    \
    --load-pretrained-mt-encoder-decoder-from $PRETRAIN