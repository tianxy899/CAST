lang=de

MODEL_DIR=./checkpoints/en-$lang
# MODEL_DIR=./checkpoints/en-$lang/st_baseline_extmt
MODEL=cress.en-de.expand.pt
# DATA=./data/en-$lang
DATA=./data/mustc/en-$lang

FAIRSEQ=./fairseq_cli

RES=./res/en-$lang/st_baseline_extmt_cress
# RES=./res/en-$lang/st_baseline_extmt

# python ./cress/scripts/average_checkpoints.py \
#   --inputs ${MODEL_DIR} --num-update-checkpoints 10 --checkpoint-upper-bound 37000 \
#   --output "${MODEL_DIR}/${MODEL}"

CUDA_VISIBLE_DEVICES=0 python $FAIRSEQ/generate.py $DATA --gen-subset tst-COMMON --task speech_to_text_modified \
    --user-dir cress --required-batch-size-multiple 1 \
    --max-tokens 4000000 --max-source-positions 4000000 --beam 8 --lenpen 1.2 \
    --config-yaml config.yaml --path ${MODEL_DIR}/${MODEL} --scoring sacrebleu --results-path ${RES} \
    1>./log/gen_tst_COMMON.log 2>&1

# tail -n 1 ${RES}/generate-tst-COMMON.txt
cat $RES/generate-tst-COMMON.txt | grep -P "^D" | sort -V | cut -f 3- > $RES/tran.$lang