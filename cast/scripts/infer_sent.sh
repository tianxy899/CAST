#!/usr/bin/env bash

lang=de
MODEL_DIR=checkpoints/en-$lang/sentence_level_st
MODEL=avg10.pt
DATA=data/en-$lang

CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA \
    --user-dir cast --required-batch-size-multiple 1 \
    --gen-subset tst-COMMON --task speech_to_text_modified \
    --max-tokens 4000000 --max-source-positions 4000000 --beam 8 --lenpen 1.2 \
    --config-yaml config.yaml --path ${MODEL_DIR}/${MODEL} --scoring sacrebleu