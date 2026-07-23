# 数据集路径（例如 ImageNet）
IMAGENET_PATH="data/imagenet64_new"

# 训练输出目录
OUTPUT_DIR="temp/64_MAR"

torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
main_fractalgen.py \
--model fractalmar_in64 --img_size 256 --num_conds 5 --class_num 2 \
--gen_bsz 128 --num_images 50000 \
--label_drop_prob 0 \
--num_iter_list 64,16 --cfg 3 --cfg_schedule linear --temperature 1.03 \
--output_dir ${OUTPUT_DIR} \
--resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} --seed 202556 --evaluate_gen \
