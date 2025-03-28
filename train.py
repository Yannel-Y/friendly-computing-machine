import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/sqwu/yolo/yolov8/ultralytics/cfg/models/v8/yolov8-faster.yaml')
    model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/root/autodl-tmp/sqwu/yolo/yolov8/cfg/TT100k.yaml',
                cache=False,
                project='runs/train',
                name='exp-v8-faster',
                epochs=100,
                batch=8,
                close_mosaic=10,
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False # close amp
                # fraction=0.2
                )