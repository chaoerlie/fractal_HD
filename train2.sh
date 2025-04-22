
IMAGENET_PATH="data/image"

OUTPUT_DIR="temp"\

export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
--master_addr=127.0.0.1 --master_port=29520 \
main_fractalgen.py \
  --model fractalar_in64 --img_size 64 --num_conds 1 \
  --batch_size 32 --eval_freq 40 --save_last_freq 10 \
  --epochs 800 --warmup_epochs 40 --class_num 1 \
  --blr 5.0e-5 --weight_decay 0.05 --attn_dropout 0.1 --proj_dropout 0.1 \
  --lr_schedule cosine --gen_bsz 32 --num_images 64 \
  --num_iter_list 64,16 --cfg 11.0 --cfg_schedule linear --temperature 1.03 \
  --output_dir ${OUTPUT_DIR} \
  --resume ${OUTPUT_DIR} \
  --data_path ${IMAGENET_PATH} --grad_checkpointing --online_eval \
  --evaluate_gen \
