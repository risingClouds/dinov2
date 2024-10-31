from dinov2.data.datasets import ImageNet
import os
import shutil
prefix = "panoramic-X-ray"
data_root = "./dinov2/data/ImageNet-1k-style"
idx = 0

def mv_func(data_root,mode="train"):
    if mode == "test":
        middle_path = mode
    else:
        middle_path = f"{mode}/{prefix}"
    global idx
    fnames = os.listdir(f"{data_root}/{middle_path}")
    for fname in fnames:
        shutil.move(
            f"{data_root}/{middle_path}/{fname}",
            f"{data_root}/{middle_path}/{prefix}_{idx}.jpg"
        )
        idx += 1

mv_func(data_root,"train")
mv_func(data_root,"val")
mv_func(data_root,"test")


for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="./dinov2/data/ImageNet-1k-style", extra="./dinov2/data/extra")
    dataset.dump_extra()