
# 数据集路径（例如 ImageNet）
IMAGENET_PATH="data/imagenet64-2"

# 训练输出目录
OUTPUT_DIR="temp/256_MAR" \

export CUDA_VISIBLE_DEVICES=0  # 或者 1，视你要用哪张 GPU 而定

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
--master_addr=127.0.0.1 --master_port=29532 \
main_fractalgen.py \
  --model fractalmar_huge_in256 --img_size 256 --num_conds 5 --guiding_pixel \
  --batch_size 1 --eval_freq 40 --save_last_freq 1 \
  --epochs 100 --warmup_epochs 40  \
  --blr 5.0e-5 --weight_decay 0.05 --attn_dropout 0.1 --proj_dropout 0.1 --lr_schedule cosine \
  --gen_bsz 256 --num_images 8000 --num_iter_list 64,16,16 --cfg 21.0 --cfg_schedule linear --temperature 1.1 \
  --output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
  --data_path ${IMAGENET_PATH} --grad_checkpointing  \
  --label_drop_prob 0 --cfg 3.0 \
  --hd_model /home/ps/zhw/fractal_HD/resnet/model_epoch_400.pth \
  --standard_hd_value 1.65 \
  --hd_weight_schedule /home/ps/zhw/fractal_HD/weight.txt \
  --mmds
  # --evaluate_gen \
