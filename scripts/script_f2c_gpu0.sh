CUDA_VISIBLE_DEVICES=0 python main_f2c_imagenet.py --f2c 0 --categories fruit_vege --data_ratio 0.8 --add_layer 0 &&
CUDA_VISIBLE_DEVICES=0 python main_f2c_imagenet.py --f2c 0 --categories dog_cat --data_ratio 0.8 --add_layer 0
