# BERT_BASE_DIR=model/wwm_uncased_L-24_H-1024_A-16
BERT_BASE_DIR=model/uncased_L-12_H-768_A-12
DATA_DIR=../data/
TASK=semre

python3 run_classifier.py \
  --task_name=$TASK \
  --do_train=true \
  --do_predict=true \
  --entity_sep=4 \
  --data_dir=$DATA_DIR/$TASK \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=105 \
  --train_batch_size=20 \
  --learning_rate=2e-5 \
  --num_train_epochs=10 \
  --output_dir=output/$TASK

