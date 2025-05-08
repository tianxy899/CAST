lang=de

# DATA=./data-en-xx/en-$lang/extra_mt/bin
DATA=data/en-$lang/extra_mt/bin
HUBERT=checkpoints/hubert_base_ls960.pt
SAVE_DIR=checkpoints/en-$lang/pretrain_mt

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train $DATA \
   --user-dir cast \
   --task speech_and_text_translation \
   --arch hubert_transformer --hubert-model-path $HUBERT \
   --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
   --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0007 \
   --criterion speech_and_text_translation --label-smoothing 0.1 --ignore-prefix-size 1 \
   --weight-decay 0.0 \
   --max-tokens 8192 --update-freq 1 \
   --no-progress-bar --log-format json --log-interval 100 \
   --save-interval-updates 5000 --keep-last-epochs 5 \
   --keep-best-checkpoints 5 \
   --save-dir $SAVE_DIR --patience 5 \
   --distributed-world-size 4 --ddp-backend=no_c10d \
   --eval-bleu --eval-bleu-args '{"beam": 4}' \
   --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
   --eval-bleu-bpe sentencepiece --eval-bleu-bpe-path data/en-$lang/spm_en$lang.model \
   --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
   --max-update 250000 \
   --report-accuracy \
   --fp16

