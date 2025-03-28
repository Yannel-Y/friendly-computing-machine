import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/sqwu/yolo/yolov8/runs/train/exp-v8-faster/weights/best.pt')
    model.val(data='/root/autodl-tmp/sqwu/yolo/yolov8/cfg/TT100k.yaml',
              batch=1,
                split='val',
                # save_json=True, # if you need to cal coco metrice
                project='runs/val',
                name='exp-v8-faster',
                )