
# 数据集路径（例如 ImageNet）
IMAGENET_PATH="data/imagenet64_new"

# 训练输出目录
OUTPUT_DIR="train/output/imagenet_256"\

export CUDA_VISIBLE_DEVICES=0,1  # 或者 1，视你要用哪张 GPU 而定

torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
--master_addr=127.0.0.1 --master_port=29532 \
main_fractalgen.py \
  --model fractalmar_large_in256 --img_size 256 --num_conds 5 --guiding_pixel \
  --batch_size 4 --eval_freq 40 --save_last_freq 10 \
  --epochs 800 --warmup_epochs 40  --class_num 2 \
  --blr 5.0e-5 --weight_decay 0.05 --attn_dropout 0.1 --proj_dropout 0.1 --lr_schedule cosine \
  --gen_bsz 256 --num_images 8000 --num_iter_list 64,16,16 --cfg 21.0 --cfg_schedule linear --temperature 1.1 \
  --output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
  --data_path ${IMAGENET_PATH} --grad_checkpointing --online_eval \
  --label_drop_prob 0 --cfg 3.0
  # --evaluate_gen \
