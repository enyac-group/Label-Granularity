CUDA_VISIBLE_DEVICES=1 python main_f2c_cifar100.py --categories animals --f2c 1 --widen_factor 2 &&
CUDA_VISIBLE_DEVICES=1 python main_f2c_cifar100.py --categories animals --f2c 1 --widen_factor 4 &&
CUDA_VISIBLE_DEVICES=1 python main_f2c_cifar100.py --categories animals --f2c 1 --widen_factor 6
