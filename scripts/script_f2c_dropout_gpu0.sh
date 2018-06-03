CUDA_VISIBLE_DEVICES=0 python main_f2c_cifar100.py --categories animals --f2c 0 --dropout 0.4 --data_ratio 1. --add_layer 0 &&
CUDA_VISIBLE_DEVICES=0 python main_f2c.py --f2c 0 --dropout 0.3 