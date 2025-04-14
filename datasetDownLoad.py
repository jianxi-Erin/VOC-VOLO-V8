import os
import requests
import tarfile
from lxml import etree
import shutil

# ----------------------
# 配置参数
# ----------------------
VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
VOC_TEST_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
DATA_DIR = "./dataset/"

# ----------------------
# 下载并解压数据集
# ----------------------
def download_and_extract(url, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
    
    filename = os.path.join(dest_dir, url.split("/")[-1])
    
    # 下载文件
    if not os.path.exists(filename):
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    
    # 解压文件
    print(f"Extracting {filename}...")
    with tarfile.open(filename) as tar:
        tar.extractall(path=dest_dir)



# 转换VOC格式到YOLO格式
# ----------------------
def convert_voc_to_yolo(voc_dir, output_dir):
    print(f"转换VOC格式到YOLO格式:{output_dir}")
    classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    # 创建输出目录
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    # 处理所有标注文件
    for split in ["trainval", "test"]:
        with open(os.path.join(voc_dir, f"VOCdevkit/VOC2007/ImageSets/Main/{split}.txt")) as f:
            ids = [line.strip() for line in f.readlines()]

        for img_id in ids:
            # 解析XML标注
            ann_path = os.path.join(voc_dir, f"VOCdevkit/VOC2007/Annotations/{img_id}.xml")
            tree = etree.parse(ann_path)
            root = tree.getroot()

            # 获取图像尺寸
            size = root.find("size")
            width = int(size.find("width").text)
            height = int(size.find("height").text)

            # 转换每个对象
            yolo_ann = []
            for obj in root.iter("object"):
                cls = obj.find("name").text
                cls_id = classes.index(cls)
                
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                
                # 转换为YOLO格式
                x_center = (xmin + xmax) / 2 / width
                y_center = (ymin + ymax) / 2 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                
                yolo_ann.append(f"{cls_id} {x_center} {y_center} {w} {h}")

            # 保存YOLO标注
            with open(os.path.join(output_dir, f"labels/{img_id}.txt"), "w") as f:
                f.write("\n".join(yolo_ann))
            
            # 复制图像（这里直接创建符号链接节省空间）
            src = os.path.join(voc_dir, f"VOCdevkit/VOC2007/JPEGImages/{img_id}.jpg")
            dst = os.path.join(output_dir, f"images/{img_id}.jpg")
            if not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)  # 如果失败则复制文件
                    # os.symlink(src, dst)  # 尝试创建符号链接
                except:
                    print(f"创建符号链接失败，复制文件: {src} -> {dst}")
    # 创建数据集配置文件
    with open("voc.yaml", "w") as f:
        names_str = '\n'.join([f"  {i}: {name}" for i, name in enumerate(classes)])
        f.write(
f"""path: {os.path.abspath(output_dir)}
train: images
val: images
test: images
names:
{names_str}
"""
                )
    
    print("转换完成！输出voc.yaml")
# # 下载训练集和测试集
download_and_extract(VOC_URL, DATA_DIR)
download_and_extract(VOC_TEST_URL, DATA_DIR)
# # 执行格式转换 voc->yolo
convert_voc_to_yolo(DATA_DIR, DATA_DIR+"/output")