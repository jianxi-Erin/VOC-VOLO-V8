# 预测输出
import os
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ------------ 全局配置 ------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "runs/detect/train/weights/best.pt"
model = YOLO(MODEL_PATH)
INPUT_PATH = "dataset/output/images/test/"  # 输入路径,可以为图片,视频,文件夹,摄像头编号
# INPUT_PATH = "dataset/output/video/test.mp4"  # 输入路径,可以为图片,视频,文件夹,摄像头编号
# INPUT_PATH=0

SAVE = True  # 是否保存预测结果
OUTPUT_PATH = "predict/"  # 预测结果保存路径

# ------------ 工具函数 ------------
def draw_boxes_pil(image, results):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = f"{model.names[cls_id]} {conf:.2f}"

        text_bbox = font.getbbox(label)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill="red")
        draw.text((x1, y1 - text_h), label, fill="white", font=font)

    return image

def save_image(image, save_path, origin_path=None):
    if os.path.isdir(save_path):
        filename = os.path.basename(origin_path)
        save_path = os.path.join(save_path, filename)
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)
    print(f"✅ 已保存图片: {save_path}")

# ------------ 单图预测 ------------
def predict_image(image_path, save=False, save_path=None):
    image = Image.open(image_path).convert("RGB")
    results = model.predict(image_path, imgsz=640, device=DEVICE)
    image = draw_boxes_pil(image, results)

    plt.imshow(image)
    plt.axis("off")
    plt.title("预测结果")
    plt.show()

    if save and save_path:
        save_image(image, save_path, origin_path=image_path)

# ------------ 视频预测 ------------
def predict_video(video_path, save=False, save_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 视频文件无法打开")
        return

    if save:
        if os.path.isdir(save_path):
            filename = os.path.basename(video_path)
            save_path = os.path.join(save_path, f"{os.path.splitext(filename)[0]}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps, w, h = cap.get(5), int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=640, device=DEVICE)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = f"{model.names[cls_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("预测中 - 按 Q 退出", frame)
        if save:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if save:
        out.release()
        print(f"✅ 已保存视频: {save_path}")
    cv2.destroyAllWindows()

# ------------ 文件夹批量图片 ------------
def predict_folder(folder_path, save=False, output_dir=None):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                img_path = os.path.join(root, file)
                image = Image.open(img_path).convert("RGB")
                results = model.predict(img_path, imgsz=640, device=DEVICE)
                image = draw_boxes_pil(image, results)

                if save and output_dir:
                    rel_path = os.path.relpath(img_path, folder_path)
                    save_path = os.path.join(output_dir, rel_path)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    image.save(save_path)

    if save:
        print(f"✅ 文件夹预测完成，结果已保存至: {output_dir}")

# ------------ 摄像头实时预测 ------------
def predict_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {index}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=640, device=DEVICE)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = f"{model.names[cls_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("摄像头预测 - 按 Q 退出", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()   
    cv2.destroyAllWindows()

# ------------ 总入口函数 ------------
def run_predict(path, save=False, save_path=None):
    if isinstance(path, int):
        predict_camera(index=path)
    elif os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            predict_image(path, save, save_path)
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            predict_video(path, save, save_path)
    elif os.path.isdir(path):
        predict_folder(path, save, save_path)
    else:
        print("❌ 无效路径，请确认输入正确的图片/视频/文件夹/摄像头编号")

# ------------ 示例调用 ------------
run_predict(INPUT_PATH, SAVE, OUTPUT_PATH)      # 预测输出
