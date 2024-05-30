python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.001 \
  --seed 0 \
  --device_id 0 \



