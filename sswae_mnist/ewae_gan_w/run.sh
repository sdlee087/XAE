export CUDA_VISIBLE_DEVICES=5
echo $CUDA_VISIBLE_DEVICES

python train.py --log_info=config/log_info.yaml --train_config=config/train_config.cfg
