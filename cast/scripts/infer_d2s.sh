lang=de

MODEL_DIR=./checkpoints/en-$lang/st_doc_gate_loss

MODEL_DIR=./checkpoints/en-$lang/st_extmt_doc_gate_loss_lambda0.3

# MODEL_DIR=./checkpoints/en-$lang/st_extmt_doc_gate_loss
# MODEL=/data/xytian/ali/CoDoST/checkpoints/en-de/st_extmt_d2s_jsd_probmix/avg10_49000_28.27.pt
# MODEL_DIR=/data/xytian/ali/CoDoST/checkpoints/en-de/st_d2s_extmt_gate_sum_avg_loss_log_nosingle_gg
MODEL=avg10_31000.pt

# DATA=./data/en-$lang/doc-data-hubert
DATA=./data/mustc/en-$lang/doc-data-hubert

FAIRSEQ=./fairseq_cli

RES=./res/en-$lang/st_extmt_doc_gate_loss

python ./cress/scripts/average_checkpoints.py \
  --inputs ${MODEL_DIR} --num-update-checkpoints 10 --checkpoint-upper-bound 31000 \
  --output "${MODEL_DIR}/${MODEL}"

CUDA_VISIBLE_DEVICES=0 python $FAIRSEQ/generate.py $DATA \
    --user-dir cress --required-batch-size-multiple 1 \
    --gen-subset tst-COMMON --task speech_to_text_modified \
    --doc-mode --max-tokens 4000000 --max-source-positions 4000000 --beam 8 --lenpen 1.2 \
    --config-yaml config.yaml --path ${MODEL_DIR}/${MODEL} --scoring sacrebleu --results-path ${RES} \
    1>./log/gen_tst_COMMON.log 2>&1

tail -n 1 ${RES}/generate-tst-COMMON.txt



# cat ${RES}/generate-tst-COMMON_st.txt | grep -P "^D" | sort -V | cut -f 3- > ${RES}/tran.${lang}

# lang=de
# REF=../res/tst-common-en-${lang}

# cat ${RES}/generate-tst-COMMON_st.txt | grep -P "^D" | sort -V | cut -f 3- > ${RES}/common_tran.${lang}

# python ../res/cat.py ${REF}/id.txt ${RES}/common_tran.${lang} ${RES}/3by3.${lang} ${RES}/doc.${lang}
# sacrebleu ${REF}/gold.${lang} -i ${RES}/common_tran.${lang} -m bleu -b -w 2
# sacrebleu ${REF}/doc.${lang} -i ${RES}/doc.${lang} -m bleu -b -w 2