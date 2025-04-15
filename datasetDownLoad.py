"""
下载 PASCAL VOC 2007 数据集并转换为 YOLO 格式,生成 voc.yaml 文件
PASCAL VOC 2007 数据集是计算机视觉领域中一个著名的标准数据集，主要用于目标检测、图像分类和语义分割等任务。
该数据集包含 9963 张图片，分为训练集（5011 张）和测试集（4952 张），
涵盖 20 个类别，如飞机、自行车、鸟、船、瓶子、公共汽车、汽车、猫、椅子、牛、餐桌、狗、马、摩托车、人、盆栽、羊、沙发、火车和电视显示器。
其标注信息以 XML 格式存储，包含目标的边界框、类别标签等。该数据集是许多经典计算机视觉模型的训练和评估基准。
"""
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
    print(f"转换VOC格式到YOLO格式: {output_dir}")

    classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    # 创建训练和测试子目录
    for split in ["train", "test"]:
        os.makedirs(os.path.join(output_dir, f"images/{split}"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, f"labels/{split}"), exist_ok=True)

    # 对 trainval 和 test 分别处理
    for split in ["trainval", "test"]:
        split_type = "train" if split == "trainval" else "test"
        list_path = os.path.join(voc_dir, f"VOCdevkit/VOC2007/ImageSets/Main/{split}.txt")
        if not os.path.exists(list_path):
            print(f"划分文件不存在: {list_path}")
            continue

        with open(list_path) as f:
            ids = [line.strip() for line in f.readlines()]
        print(f"开始处理 {split_type} 集，共 {len(ids)} 张图片")

        for img_id in ids:
            ann_path = os.path.join(voc_dir, f"VOCdevkit/VOC2007/Annotations/{img_id}.xml")
            if not os.path.exists(ann_path):
                print(f"标注文件不存在: {ann_path}")
                continue

            tree = etree.parse(ann_path)
            root = tree.getroot()

            size = root.find("size")
            width = int(size.find("width").text)
            height = int(size.find("height").text)

            yolo_ann = []
            for obj in root.iter("object"):
                cls = obj.find("name").text
                if cls not in classes:
                    continue
                cls_id = classes.index(cls)

                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)

                x_center = (xmin + xmax) / 2 / width
                y_center = (ymin + ymax) / 2 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height

                yolo_ann.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            # 保存 YOLO 标签
            label_out_path = os.path.join(output_dir, f"labels/{split_type}/{img_id}.txt")
            with open(label_out_path, "w") as f:
                f.write("\n".join(yolo_ann))

            # 拷贝图像
            src = os.path.join(voc_dir, f"VOCdevkit/VOC2007/JPEGImages/{img_id}.jpg")
            dst = os.path.join(output_dir, f"images/{split_type}/{img_id}.jpg")
            if not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                    # os.symlink(src, dst)  # 尝试创建符号链接                    
                except:
                    print(f"复制图像失败: {src}")

    # 创建 voc.yaml 文件
    with open("voc.yaml", "w") as f:
        names_str = '\n'.join([f"  {i}: {name}" for i, name in enumerate(classes)])
        f.write(
f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/train
test: images/test
names:
{names_str}
"""
        )
    print("✅ VOC 转 YOLO 完成，生成 voc.yaml")


# # 下载并解压训练集和测试集
download_and_extract(VOC_URL, DATA_DIR)
download_and_extract(VOC_TEST_URL, DATA_DIR)
# # 执行格式转换 voc->yolo
convert_voc_to_yolo(DATA_DIR, DATA_DIR+"/output")