import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov8', choices=['yolov8', 'yolov8-faster'], help='选择使用的模型')
    args = parser.parse_args()

    if args.model == 'yolov8':
        model_path = './runs/train/exp-v8/weights/best.pt'  # YOLOv8 模型路径
    else:
        model_path = './runs/train/exp-v8-faster/weights/best.pt'  # YOLOv8-Faster 模型路径

    model = YOLO(model_path)  # 加载选定的模型
    model.to('cpu')  # 将模型移动到 CPU
    model.predict(source='./data/73.jpg',  # 替换为你的单张图像路径
                  project='runs/detect',
                  name='exp',
                  save=True,  # 保存检测的最终结果
                  )  # 不需要 visualize