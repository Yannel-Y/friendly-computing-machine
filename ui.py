import sys
import warnings
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from ultralytics import YOLO

warnings.filterwarnings('ignore')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("交通标志检测系统")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(400, 300)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.result_label = QLabel(self)
        self.result_label.setFixedSize(400, 300)
        self.result_label.setAlignment(Qt.AlignCenter)

        self.select_image_button = QPushButton("选择图像", self)
        self.select_image_button.clicked.connect(self.select_image)

        self.select_model_button = QPushButton("选择模型", self)
        self.select_model_button.clicked.connect(self.select_model)

        self.detect_button = QPushButton("检测", self)
        self.detect_button.clicked.connect(self.detect)

        self.model_path = './runs/train/exp-v8/weights/best.pt'  # 默认使用 YOLOv8 模型

        layout = QVBoxLayout(self.central_widget)
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.result_label)
        layout.addLayout(image_layout)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.select_image_button)
        button_layout.addWidget(self.select_model_button)
        button_layout.addWidget(self.detect_button)
        layout.addLayout(button_layout)

    def select_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)
            self.image_path = file_name

    def select_model(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择模型", "", "PyTorch Models (*.pt);;All Files (*)", options=options)
        if file_name:
            self.model_path = file_name

    def detect(self):
        model = YOLO(self.model_path)
        model.to('cpu')
        model.predict(source=self.image_path,
                      project='runs/detect',
                      name='exp',
                      save=True)

        result_path = 'runs/detect/exp/' + self.image_path.split('/')[-1]
        pixmap = QPixmap(result_path)
        scaled_pixmap = pixmap.scaled(self.result_label.width(), self.result_label.height(), Qt.KeepAspectRatio)
        self.result_label.setPixmap(scaled_pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())