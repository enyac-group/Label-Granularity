CUDA_VISIBLE_DEVICES=0 python main_f2c_cifar100.py --f2c 0 --data_ratio 0.8 --add_layer 0 &&
CUDA_VISIBLE_DEVICES=0 python main_f2c_cifar100.py --categories animals --f2c 1 --data_ratio 0.8 --add_layer 0 
