# 数据集路径（请确保替换为512x512或更高分辨率的数据集）
IMAGENET_PATH="data/imagenet64_new"

# 训练输出目录
OUTPUT_DIR="train/output/imagenet_512"

export CUDA_VISIBLE_DEVICES=0,1  # 根据您的GPU数量调整

# 由于模型尺寸大幅增加，建议使用多GPU并行训练
# torchrun --nproc_per_node=<GPU数量>
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
--master_addr=127.0.0.1 --master_port=29523 \
main_fractalgen.py \
  --model fractalmar_huge_in512 --img_size 512 --num_conds 1 \
  --batch_size 2 --eval_freq 40 --save_last_freq 10 \
  --epochs 800 --warmup_epochs 40 --class_num 2 \
  --blr 2.5e-5 --weight_decay 0.05 --attn_dropout 0.1 --proj_dropout 0.1 \
  --lr_schedule cosine --gen_bsz 4 --num_images 64 \
  --num_iter_list 64,16,16,16  --cfg_schedule linear --temperature 1.03 \
  --output_dir ${OUTPUT_DIR} \
  --resume ${OUTPUT_DIR} \
  --data_path ${IMAGENET_PATH} --grad_checkpointing --online_eval \
  --label_drop_prob 0.1 --cfg 3.0
  # --evaluate_gen
