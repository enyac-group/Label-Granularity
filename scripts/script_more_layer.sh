CUDA_VISIBLE_DEVICES=0 python main_f2c.py --f2c 1 --data_ratio 1. &&
CUDA_VISIBLE_DEVICES=0 python main_f2c_cifar100.py --f2c 1 --data_ratio 1. --categories animals &&
CUDA_VISIBLE_DEVICES=0 python main_f2c_cifar100.py --f2c 1 --data_ratio 1. &&
CUDA_VISIBLE_DEVICES=0 python main_f2c_imagenet.py --f2c 1 --categories fruit_vege --data_ratio 1. &&
CUDA_VISIBLE_DEVICES=0 python main_f2c_imagenet.py --f2c 1 --categories dog_cat --data_ratio 1.
