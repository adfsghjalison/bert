BERT_BASE_DIR=model/wwm_uncased_L-24_H-1024_A-16
DATA_DIR=data/
TASK=semre


CUDA_VISIBLE_DEVICES=1 python3 run_classifier.py \
  --task_name=$TASK \
  --do_train=false \
  --do_predict=true \
  --data_dir=$DATA_DIR/$TASK \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=110 \
  --train_batch_size=12 \
  --learning_rate=2e-5 \
  --num_train_epochs=1.0 \
  --output_dir=output/$TASK

