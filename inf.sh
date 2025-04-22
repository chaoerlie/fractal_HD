# 数据集路径（例如 ImageNet）
IMAGENET_PATH="data/image"

# 训练输出目录
OUTPUT_DIR="temp"

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
main_fractalgen.py \
--model fractalar_in64 --img_size 64 --num_conds 1 --class_num 1000 \
--gen_bsz 32 --num_images 10000 \
--num_iter_list 64,16 --cfg 11.0 --cfg_schedule linear --temperature 1.03 \
--output_dir temp \
--resume temp \
--data_path ${IMAGENET_PATH} --seed 0 --evaluate_gen