export PYTHONPATH=$PYTHONPATH:$(pwd)    

python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/train/vitl14_panoramic.yaml \
    --output-dir ./output/vitl14_panoramic \
    train.dataset_path=ImageNet:split=TRAIN:root=dinov2/data/ImageNet-1k-style:extra=dinov2/data/extra