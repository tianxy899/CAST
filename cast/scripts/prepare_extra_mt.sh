#!/usr/bin/env bash

TGT_LANG=fr
TEXT_PATH=./data/en-$TGT_LANG/extra_mt

spm_model=./data/en-$TGT_LANG/spm_en$TGT_LANG.model
spm_dict=./data/en-$TGT_LANG/spm_en$TGT_LANG.txt
# train data
python3 ./cress/prepare_data/apply_spm.py --model ${spm_model} --input-file ${TEXT_PATH}/train.${TGT_LANG} --output-file ${TEXT_PATH}/train.spm.${TGT_LANG} --add_lang_tag ${TGT_LANG}
python3 ./cress/prepare_data/apply_spm.py --model ${spm_model} --input-file ${TEXT_PATH}/train.en --output-file ${TEXT_PATH}/train.spm.en --add_lang_tag en
# dev data
python3 ./cress/prepare_data/apply_spm.py --model ${spm_model} --input-file ${TEXT_PATH}/dev.${TGT_LANG} --output-file ${TEXT_PATH}/dev.spm.${TGT_LANG} --add_lang_tag ${TGT_LANG}
python3 ./cress/prepare_data/apply_spm.py --model ${spm_model} --input-file ${TEXT_PATH}/dev.en --output-file ${TEXT_PATH}/dev.spm.en --add_lang_tag en

fairseq-preprocess \
    --source-lang en --target-lang ${TGT_LANG} \
    --trainpref ${TEXT_PATH}/train.spm --validpref ${TEXT_PATH}/dev.spm \
    --destdir ${TEXT_PATH}/bin --thresholdtgt 0 --thresholdsrc 0 \
    --srcdict ${spm_dict} --tgtdict ${spm_dict} \
    --workers 100