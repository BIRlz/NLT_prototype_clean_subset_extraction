
# warm-up train model
CUDA_VISIBLE_DEVICES=0 python NLT_main_warmup.py --dataset cifar10 --imb_factor 0.01 --noise_ratio 0.1
CUDA_VISIBLE_DEVICES=1 python NLT_main_warmup.py --dataset cifar10 --imb_factor 0.01 --noise_ratio 0.2
CUDA_VISIBLE_DEVICES=2 python NLT_main_warmup.py --dataset cifar10 --imb_factor 0.01 --noise_ratio 0.3
CUDA_VISIBLE_DEVICES=3 python NLT_main_warmup.py --dataset cifar10 --imb_factor 0.01 --noise_ratio 0.4

# # main different beta on res18
# CUDA_VISIBLE_DEVICES=0 python NLT_main.py --dataset cifar100 --beta_flag None 
# CUDA_VISIBLE_DEVICES=1 python NLT_main.py --dataset cifar100 --beta_flag 0.9
# CUDA_VISIBLE_DEVICES=2 python NLT_main.py --dataset cifar100 --beta_flag 0.95
# CUDA_VISIBLE_DEVICES=3 python NLT_main.py --dataset cifar100 --beta_flag 0.98

# # Ablation study
# CUDA_VISIBLE_DEVICES=0 python NLT_ablation.py --dataset cifar10 --beta_flag 0.98
# CUDA_VISIBLE_DEVICES=1 python NLT_ablation.py --dataset cifar10 --beta_flag 0.95
# CUDA_VISIBLE_DEVICES=2 python NLT_ablation.py --dataset cifar10 --beta_flag 0.90
# CUDA_VISIBLE_DEVICES=3 python NLT_ablation.py --dataset cifar10 

# CUDA_VISIBLE_DEVICES=0 python NLT_ablation.py --dataset cifar100 --data_split tabasco --imb_factor 0.1 --noise_ratio 0.5
# CUDA_VISIBLE_DEVICES=1 python NLT_ablation.py --dataset cifar100 --beta_flag 0.95
# CUDA_VISIBLE_DEVICES=2 python NLT_ablation.py --dataset cifar100 --beta_flag 0.90


# CUDA_VISIBLE_DEVICES=0 python NLT_ablation.py --dataset cifar100 --imb_factor 0.01 --noise_ratio 0.5 --lam 0
# CUDA_VISIBLE_DEVICES=1 python NLT_ablation.py --dataset cifar100 --imb_factor 0.01 --noise_ratio 0.5 --lam 0.1
# CUDA_VISIBLE_DEVICES=2 python NLT_ablation.py --dataset cifar100 --imb_factor 0.01 --noise_ratio 0.5 --lam 0.3
# CUDA_VISIBLE_DEVICES=3 python NLT_ablation.py --dataset cifar100 --imb_factor 0.01 --noise_ratio 0.5 --lam 0.5
# CUDA_VISIBLE_DEVICES=0 python NLT_ablation.py --dataset cifar100 --imb_factor 0.01 --noise_ratio 0.5 --lam 0.7
# CUDA_VISIBLE_DEVICES=1 python NLT_ablation.py --dataset cifar100 --imb_factor 0.01 --noise_ratio 0.5 --lam 1

# # RoLT
# CUDA_VISIBLE_DEVICES=0 python Train_cifar.py --dataset cifar100 --imb_factor 0.1 --noise_ratio 0.5 
# CUDA_VISIBLE_DEVICES=1 python Train_cifar.py --dataset cifar100 --imb_factor 0.01 --noise_ratio 0.5
# CUDA_VISIBLE_DEVICES=2 python Train_cifar.py --dataset cifar10 --imb_factor 0.1 --noise_ratio 0.5
# CUDA_VISIBLE_DEVICES=3 python Train_cifar.py --dataset cifar10 --imb_factor 0.01 --noise_ratio 0.5

# # label transfer
# CUDA_VISIBLE_DEVICES=0 python NLT_transfer.py --dataset cifar10 --imb_factor 0.1 --noise_ratio 0.3
# CUDA_VISIBLE_DEVICES=1 python NLT_transfer.py --dataset cifar10 --imb_factor 0.1 --noise_ratio 0.5
# CUDA_VISIBLE_DEVICES=2 python NLT_transfer.py --dataset cifar10 --imb_factor 0.01 --noise_ratio 0.3
# CUDA_VISIBLE_DEVICES=0 python NLT_transfer.py --dataset cifar10 --imb_factor 0.01 --noise_ratio 0.5

# CUDA_VISIBLE_DEVICES=0 python NLT_transfer.py --dataset cifar100 --imb_factor 0.1 --noise_ratio 0.3
# CUDA_VISIBLE_DEVICES=1 python NLT_transfer.py --dataset cifar100 --imb_factor 0.1 --noise_ratio 0.5
# CUDA_VISIBLE_DEVICES=2 python NLT_transfer.py --dataset cifar100 --imb_factor 0.01 --noise_ratio 0.3
# CUDA_VISIBLE_DEVICES=3 python NLT_transfer.py --dataset cifar100 --imb_factor 0.01 --noise_ratio 0.5

# # compare with tabasco
# CUDA_VISIBLE_DEVICES=0 python NLT_main.py --dataset cifar100 --data_split tabasco 
# CUDA_VISIBLE_DEVICES=1 python NLT_main.py --dataset cifar100 --data_split tabasco
# CUDA_VISIBLE_DEVICES=2 python NLT_main.py --dataset cifar100 --data_split tabasco
# CUDA_VISIBLE_DEVICES=3 python NLT_main.py --dataset cifar100 --data_split tabasco

# # main on resre
# CUDA_VISIBLE_DEVICES=0 python NLT_main_res32.py --dataset cifar10 
# CUDA_VISIBLE_DEVICES=1 python NLT_main_res32.py --dataset cifar10 
# CUDA_VISIBLE_DEVICES=2 python NLT_main_res32.py --dataset cifar10 
# CUDA_VISIBLE_DEVICES=3 python NLT_main_res32.py --dataset cifar10 
