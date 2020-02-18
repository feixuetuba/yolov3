#从coco数据集中抽取特定label的样本
import sys
import shutil
import os
sys.path.insert(0, ".")
from argparse import ArgumentParser
from utils.utils import load_classes

def extract_samples(src, dest, targets=[]):
    train_fd = open(os.path.join(dest, "train.txt"), "w")
    val_fd = open(os.path.join(dest, "val.txt"), "w")
    classes = load_classes("./data/coco.names")
    if len(targets) == 0:
        targets = [str(i) for i in range(80)]
    else:
        targets = [str(classes.index(_)) for _ in targets]
    for dataset, record_f in zip(["train2014", "val2014"], [train_fd, val_fd]):
        os.makedirs(os.path.join(dest, dataset, "labels"), exist_ok=True)
        os.makedirs(os.path.join(dest, dataset, "images"), exist_ok=True)
        for f in os.listdir("%s/%s"%(src, dataset)):
            label_file = os.path.join(src, dataset, f)
            with open(label_file, "r") as fd:
                lines = fd.readlines()
            annos = []

            for line in lines:
                cls, x, y, w, h = line.strip().split(" ")
                if cls in targets:
                    annos.append(line)
            if len(annos) > 0:
                dest_label = os.path.join(dest, dataset, "labels", f)
                with open(dest_label, "w") as fw:
                    for line in annos:
                        fw.write(line)
                src_img = label_file.replace("labels", "images").replace(".txt", ".jpg")
                dest_img = dest_label.replace("labels", "images").replace(".txt", ".jpg")
                shutil.copy(src_img, dest_img)
                record_f.write(dest_label)

def get_ops():
    parser = ArgumentParser()
    parser.add_argument("-s", "--src")
    parser.add_argument("-o", "--output")
    parser.add_argument("-t", "--targets", type=str, help="split by '，'")
    return  parser.parse_args()

if __name__ == '__main__':
    opt = get_ops()
    extract_samples(opt.src, opt.output, opt.targets.split(","))