



torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
main_fractalgen.py \
--model fractalmar_huge_in256 --img_size 256 --num_conds 5 --guiding_pixel \
--gen_bsz 128 --num_images 1000 \
--num_iter_list 64,16,16 --cfg 19.0 --cfg_schedule linear --temperature 1.1 \
--output_dir /home/ps/zhw/fractal_HD/temp/256_MAR \
--resume /home/ps/zhw/fractal_HD/temp/256_MAR \
--data_path /home/ps/zhw/fractal_HD/data/imagenet64_new --seed 20251008 --evaluate_gen