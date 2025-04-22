
# 数据集路径（例如 ImageNet）
IMAGENET_PATH="data/fractal"

# 训练输出目录
OUTPUT_DIR="train/output/fractalgen_run"\


# torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
# --master_addr=127.0.0.1 --master_port=29510 \
# main_fractalgen.py \
# --model fractalar_in64 --img_size 64 --num_conds 1 \
# --batch_size 1 --eval_freq 40 --save_last_freq 10 \
# --epochs 800 --warmup_epochs 40 --class_num 1 \
# --blr 5.0e-5 --weight_decay 0.05 --attn_dropout 0.1 --proj_dropout 0.1 --lr_schedule cosine \
# --gen_bsz 8 --num_images 50 --num_iter_list 64,16 --cfg 11.0 --cfg_schedule linear --temperature 1.03 \
# --output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
# --data_path ${IMAGENET_PATH} --grad_checkpointing --online_eval

export CUDA_VISIBLE_DEVICES=0  # 或者 1，视你要用哪张 GPU 而定

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
--master_addr=127.0.0.1 --master_port=29500 \
main_fractalgen.py \
  --model fractalar_in64 --img_size 64 --num_conds 1 \
  --batch_size 32 --eval_freq 40 --save_last_freq 10 \
  --epochs 1600 --warmup_epochs 40 --class_num 2 \
  --blr 5.0e-5 --weight_decay 0.05 --attn_dropout 0.1 --proj_dropout 0.1 \
  --lr_schedule cosine --gen_bsz 16 --num_images 64 \
  --num_iter_list 64,16 --cfg 11.0 --cfg_schedule linear --temperature 1.03 \
  --output_dir ${OUTPUT_DIR} \
  --resume ${OUTPUT_DIR} \
  --data_path ${IMAGENET_PATH} --grad_checkpointing --online_eval \
  # --evaluate_gen \
