#!/usr/bin/evn python
# coding:utf-8
import os
import shutil

from argparse import ArgumentParser

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import sys


def filter(labels, src, dest):
    tree = ET.parse(src)
    root = tree.getroot()  # 获得root节点
    valid = False
    for object in root.findall('object'):
        name = object.find("name").text
        if name not in labels:
            root.remove(object)
            continue
        valid = True
    if valid:
        tree.write(dest)
    return valid 

def get_opts():
    parser = ArgumentParser()
    parser.add_argument("-s", "--src", help="src xml/dir")
    parser.add_argument("-d", "--dest", help="dest xml/dir")
    parser.add_argument("-t", "--type", default = "voc", help="data set type")
    parser.add_argument("-l", "--labels", help="labels required, split by ','")
    return parser.parse_args()

def voc_filter(lables, base_dir, dest_dir):
    for sub in ["VOC2007", "VOC2012"]:
        src_images = os.path.join(base_dir, sub, "JPEGImages")
        dest_images = os.path.join(dest_dir, "JPEGImages")
        dest_labels = os.path.join(dest_dir, "Annotations")

        os.makedirs(dest_images, exist_ok=True)
        os.makedirs(dest_labels, exist_ok=True)

        for _ in os.listdir(src_images):
            src_image = os.path.join(src_images, _)
            src_label = src_image.replace("JPEGImages", "Annotations").replace(".jpg", ".xml")
            dest_image = os.path.join(dest_images, "%s_%s"%(sub,_))
            dest_label = os.path.join(dest_dir, "Annotations", "%s_%s"%(sub,_.replace(".jpg", ".xml")))
            if filter(labels, src_label, dest_label):
                shutil.copy(src_image, dest_image)

def comm_filter(lables, base_dir, dest_dir):
    src_images = os.path.join(base_dir, "JPEGImages")
    dest_images = os.path.join(dest_dir, "JPEGImages")
    dest_labels = os.path.join(dest_dir, "Annotations")

    os.makedirs(dest_images, exist_ok=True)
    os.makedirs(dest_labels, exist_ok=True)

    for _ in os.listdir(src_images):
        src_image = os.path.join(src_images, _)
        src_label = src_image.replace("JPEGImages", "Annotations").replace(".jpg", ".xml")
        dest_image = os.path.join(dest_images, _)
        dest_label = os.path.join(dest_dir, "Annotations", _.replace(".jpg", ".xml"))
        if filter(labels, src_label, dest_label):
            shutil.copy(src_image, dest_image)

if __name__ == "__main__":
    opt = get_opts()
    labels = opt.labels.split(",")
    src = opt.src
    dest = opt.dest

    assert os.path.isdir(src), "src should be the root of VOC dataset"
    assert dest != src, "dest and src should not be the same!"
    assert len(labels) != 0, "labels required should not ben empty"
    if opt.type == "voc":
        voc_filter(labels, src, dest)
    else:
        comm_filter(labels, src, dest)
