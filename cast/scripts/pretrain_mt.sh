lang=de

# DATA=./data-en-xx/en-$lang/extra_mt/bin
DATA=/data/xytian/ali/CoDoST/data-en-xx/en-$lang/en-de-sent/extra_mt/bin
HUBERT=/data/xytian/ali/CoDoST/checkpoints/hubert_base_ls960.pt
SAVE_DIR=/data/xytian/ali/CoDoST/checkpoints/en-$lang/pretrain_mt

CUDA_VISIBLE_DEVICES=4,5,6,7 fairseq-train $DATA \
   --user-dir /data/xytian/ali/CoDoST/cast \
   --task translation_with_langtag --lang-prefix-tok='<lang:'$lang'>' \
   --arch hubert_transformer --hubert-model-path $HUBERT \
   --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
   --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0007 \
   --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --ignore-prefix-size 1 \
   --weight-decay 0.0 \
   --max-tokens 8192 --update-freq 1 \
   --no-progress-bar --log-format json --log-interval 100 \
   --save-interval-updates 5000 --keep-last-epochs 10 \
   --keep-best-checkpoints 5 \
   --save-dir $SAVE_DIR \
   --distributed-world-size 4 --ddp-backend=no_c10d \
   --eval-bleu --eval-bleu-args '{"beam": 4}' \
   --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
   --eval-bleu-bpe sentencepiece --eval-bleu-bpe-path ./data-en-xx/en-$lang/spm_en$lang.model \
   --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
   --max-update 250000 \
   --report-accuracy \
   --fp16 1>./log/en-$lang/pretrain_mt.log 2>&1

