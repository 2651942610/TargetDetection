import os
import sys
import platform
import torch
import cv2
import numpy as np
from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QRect
from PyQt5.QtGui import QIcon, QPixmap, QMovie, QPalette, QColor, QFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTabWidget, QLabel, QPushButton, 
                           QFileDialog, QMessageBox, QFrame, QGridLayout,
                           QProgressBar, QTextEdit, QSplitter)

import detect
from utils.myutil import file_is_pic, Globals


class ModernButton(QPushButton):
    """现代化风格按钮"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                border: none;
                color: white;
                padding: 12px 24px;
                text-align: center;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #357ABD;
                transform: translateY(-2px);
            }
            QPushButton:pressed {
                background-color: #2E6DA4;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)


class ImageDisplayWidget(QLabel):
    """图像显示组件"""
    def __init__(self, placeholder_text="点击加载图片", parent=None):
        super().__init__(parent)
        self.placeholder_text = placeholder_text
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #CCCCCC;
                border-radius: 12px;
                background-color: #F8F9FA;
                color: #6C757D;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        self.setText(placeholder_text)
        self.setScaledContents(False)  # 禁用自动缩放，手动控制宽高比
    
    def set_image(self, image_path):
        """设置显示图片"""
        if image_path and os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            self.set_pixmap_with_aspect_ratio(pixmap)
        else:
            self.clear()
            self.setText(self.placeholder_text)
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #CCCCCC;
                    border-radius: 12px;
                    background-color: #F8F9FA;
                    color: #6C757D;
                    font-size: 16px;
                    font-weight: bold;
                }
            """)
    
    def set_pixmap_with_aspect_ratio(self, pixmap):
        """设置pixmap并保持宽高比"""
        if pixmap and not pixmap.isNull():
            # 按比例缩放，保持宽高比
            scaled_pixmap = pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            super().setPixmap(scaled_pixmap)
            self.setStyleSheet("""
                QLabel {
                    border: 2px solid #4A90E2;
                    border-radius: 12px;
                    background-color: white;
                }
            """)
    
    def setPixmap(self, pixmap):
        """重写setPixmap方法以始终保持宽高比"""
        if pixmap and not pixmap.isNull() and self.size().width() > 0 and self.size().height() > 0:
            # 按比例缩放，保持宽高比
            scaled_pixmap = pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            super().setPixmap(scaled_pixmap)
        else:
            super().setPixmap(pixmap)


class VideoDisplayWidget(QLabel):
    """视频显示组件"""
    def __init__(self, placeholder_text="点击加载视频", parent=None):
        super().__init__(parent)
        self.placeholder_text = placeholder_text
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #CCCCCC;
                border-radius: 12px;
                background-color: #F8F9FA;
                color: #6C757D;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        self.setText(placeholder_text)
        self.setScaledContents(True)
        
        # 视频播放相关
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
    
    def load_video(self, video_path):
        """加载视频"""
        if video_path and os.path.exists(video_path):
            self.cap = cv2.VideoCapture(video_path)
            if self.cap.isOpened():
                self.timer.start(30)  # 30ms间隔播放
                self.setStyleSheet("""
                    QLabel {
                        border: 2px solid #4A90E2;
                        border-radius: 12px;
                        background-color: white;
                    }
                """)
                return True
        return False
    
    def update_frame(self):
        """更新视频帧"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.setPixmap(pixmap)
            else:
                self.stop_video()
    
    def stop_video(self):
        """停止视频播放"""
        if self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None


class ModelManager:
    """模型管理器"""
    def __init__(self):
        self.models = {
            'yolov5n': 'resource/yolov5n.pt',
            'yolov5s': 'resource/yolov5s.pt', 
            'yolov5m': 'resource/yolov5m.pt',
            'yolov5l': 'resource/yolov5l.pt',
            'yolov5x': 'resource/yolov5x.pt'
        }
        self.selected_model = self.auto_select_model()
    
    def auto_select_model(self):
        """根据设备性能自动选择模型"""
        # 检查CUDA是否可用
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            if gpu_memory >= 8 and gpu_count >= 1:
                return self.models['yolov5x']
            elif gpu_memory >= 6:
                return self.models['yolov5l']
            elif gpu_memory >= 4:
                return self.models['yolov5m']
            else:
                return self.models['yolov5s']
        else:
            # CPU模式，选择较轻量的模型
            return self.models['yolov5n']
    
    def get_model_info(self):
        """获取当前模型信息"""
        model_name = None
        for name, path in self.models.items():
            if path == self.selected_model:
                model_name = name
                break
        
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        return f"当前模型: {model_name} | 设备: {device}"


class ImageDetectionTab(QWidget):
    """图像检测选项卡"""
    def __init__(self, model_manager):
        super().__init__()
        self.model_manager = model_manager
        self.image_path = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 标题
        title = QLabel("图像目标检测")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2C3E50;
                margin: 20px;
            }
        """)
        layout.addWidget(title)
        
        # 图像显示区域
        image_layout = QHBoxLayout()
        
        # 原图显示
        left_frame = QFrame()
        left_frame.setStyleSheet("QFrame { background-color: white; border-radius: 12px; }")
        left_layout = QVBoxLayout(left_frame)
        
        original_label = QLabel("原始图像")
        original_label.setAlignment(Qt.AlignCenter)
        original_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #495057; margin: 10px;")
        left_layout.addWidget(original_label)
        
        self.original_image = ImageDisplayWidget("点击加载图片")
        left_layout.addWidget(self.original_image)
        
        # 检测结果显示
        right_frame = QFrame()
        right_frame.setStyleSheet("QFrame { background-color: white; border-radius: 12px; }")
        right_layout = QVBoxLayout(right_frame)
        
        result_label = QLabel("检测结果")
        result_label.setAlignment(Qt.AlignCenter)
        result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #495057; margin: 10px;")
        right_layout.addWidget(result_label)
        
        self.result_image = ImageDisplayWidget("检测结果将在此显示")
        right_layout.addWidget(self.result_image)
        
        image_layout.addWidget(left_frame)
        image_layout.addWidget(right_frame)
        layout.addLayout(image_layout)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.load_btn = ModernButton("📁 加载图片")
        self.load_btn.clicked.connect(self.load_image)
        
        self.detect_btn = ModernButton("🔍 开始检测")
        self.detect_btn.clicked.connect(self.detect_image)
        self.detect_btn.setEnabled(False)
        
        button_layout.addStretch()
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.detect_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # 状态信息
        self.status_label = QLabel(self.model_manager.get_model_info())
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #6C757D; font-size: 12px; margin: 10px;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def load_image(self):
        """加载图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", 
            "Image files (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        
        if file_path:
            self.image_path = file_path
            self.original_image.set_image(file_path)
            self.detect_btn.setEnabled(True)
            self.result_image.set_image(None)  # 清空结果
    
    def detect_image(self):
        """检测图片"""
        if not self.image_path:
            return
        
        self.detect_btn.setEnabled(False)
        self.detect_btn.setText("🔄 检测中...")
        
        # 创建检测线程
        self.detection_thread = ImageDetectionThread(
            self.image_path, 
            self.model_manager.selected_model
        )
        self.detection_thread.finished.connect(self.on_detection_finished)
        self.detection_thread.start()
    
    def on_detection_finished(self, result_path):
        """检测完成回调"""
        self.detect_btn.setEnabled(True)
        self.detect_btn.setText("🔍 开始检测")
        
        if result_path and os.path.exists(result_path):
            self.result_image.set_image(result_path)
            QMessageBox.information(self, "检测完成", f"检测结果已保存到: {result_path}")
        else:
            QMessageBox.warning(self, "检测失败", "检测过程中出现错误")


class VideoDetectionTab(QWidget):
    """视频检测选项卡"""
    def __init__(self, model_manager):
        super().__init__()
        self.model_manager = model_manager
        self.video_path = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 标题
        title = QLabel("视频目标检测")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2C3E50;
                margin: 20px;
            }
        """)
        layout.addWidget(title)
        
        # 视频显示区域
        video_layout = QHBoxLayout()
        
        # 原视频显示
        left_frame = QFrame()
        left_frame.setStyleSheet("QFrame { background-color: white; border-radius: 12px; }")
        left_layout = QVBoxLayout(left_frame)
        
        original_label = QLabel("原始视频")
        original_label.setAlignment(Qt.AlignCenter)
        original_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #495057; margin: 10px;")
        left_layout.addWidget(original_label)
        
        self.original_video = VideoDisplayWidget("点击加载视频")
        left_layout.addWidget(self.original_video)
        
        # 检测结果显示
        right_frame = QFrame()
        right_frame.setStyleSheet("QFrame { background-color: white; border-radius: 12px; }")
        right_layout = QVBoxLayout(right_frame)
        
        result_label = QLabel("检测结果")
        result_label.setAlignment(Qt.AlignCenter)
        result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #495057; margin: 10px;")
        right_layout.addWidget(result_label)
        
        self.result_video = VideoDisplayWidget("检测结果将在此显示")
        right_layout.addWidget(self.result_video)
        
        video_layout.addWidget(left_frame)
        video_layout.addWidget(right_frame)
        layout.addLayout(video_layout)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.load_btn = ModernButton("📁 加载视频")
        self.load_btn.clicked.connect(self.load_video)
        
        self.detect_btn = ModernButton("🔍 开始检测")
        self.detect_btn.clicked.connect(self.detect_video)
        self.detect_btn.setEnabled(False)
        
        button_layout.addStretch()
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.detect_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # 状态信息
        self.status_label = QLabel(self.model_manager.get_model_info())
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #6C757D; font-size: 12px; margin: 10px;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def load_video(self):
        """加载视频"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", 
            "Video files (*.mp4 *.avi *.mov *.mkv *.flv)"
        )
        
        if file_path:
            self.video_path = file_path
            if self.original_video.load_video(file_path):
                self.detect_btn.setEnabled(True)
                self.result_video.stop_video()  # 清空结果
    
    def detect_video(self):
        """检测视频"""
        if not self.video_path:
            return
        
        self.detect_btn.setEnabled(False)
        self.detect_btn.setText("🔄 检测中...")
        
        # 创建检测线程
        self.detection_thread = VideoDetectionThread(
            self.video_path, 
            self.model_manager.selected_model
        )
        self.detection_thread.finished.connect(self.on_detection_finished)
        self.detection_thread.start()
    
    def on_detection_finished(self, result_path):
        """检测完成回调"""
        self.detect_btn.setEnabled(True)
        self.detect_btn.setText("🔍 开始检测")
        
        if result_path and os.path.exists(result_path):
            self.result_video.load_video(result_path)
            QMessageBox.information(self, "检测完成", f"检测结果已保存到: {result_path}")
        else:
            QMessageBox.warning(self, "检测失败", "检测过程中出现错误")


class CameraDetectionTab(QWidget):
    """摄像头检测选项卡"""
    def __init__(self, model_manager):
        super().__init__()
        self.model_manager = model_manager
        self.is_detecting = False
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 标题
        title = QLabel("摄像头实时检测")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2C3E50;
                margin: 20px;
            }
        """)
        layout.addWidget(title)
        
        # 摄像头显示区域
        camera_frame = QFrame()
        camera_frame.setStyleSheet("QFrame { background-color: white; border-radius: 12px; }")
        camera_layout = QVBoxLayout(camera_frame)
        
        camera_label = QLabel("实时检测画面")
        camera_label.setAlignment(Qt.AlignCenter)
        camera_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #495057; margin: 10px;")
        camera_layout.addWidget(camera_label)
        
        self.camera_display = ImageDisplayWidget("点击开始摄像头检测")
        self.camera_display.setMinimumSize(640, 480)
        camera_layout.addWidget(self.camera_display)
        
        layout.addWidget(camera_frame)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.start_btn = ModernButton("📹 开始检测")
        self.start_btn.clicked.connect(self.start_detection)
        
        self.stop_btn = ModernButton("⏹ 停止检测")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        
        button_layout.addStretch()
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # 状态信息
        self.status_label = QLabel(self.model_manager.get_model_info())
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #6C757D; font-size: 12px; margin: 10px;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def start_detection(self):
        """开始摄像头检测"""
        if not self.is_detecting:
            self.is_detecting = True
            Globals.camera_running = True
            
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            
            # 创建摄像头检测线程
            self.camera_thread = CameraDetectionThread(
                self.model_manager.selected_model,
                self.camera_display
            )
            self.camera_thread.start()
    
    def stop_detection(self):
        """停止摄像头检测"""
        if self.is_detecting:
            self.is_detecting = False
            Globals.camera_running = False
            
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            
            self.camera_display.set_image(None)
            
            # 检查result目录中是否有新的检测结果
            result_dir = "result"
            if os.path.exists(result_dir):
                result_files = []
                for root, dirs, files in os.walk(result_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi')):
                            result_files.append(os.path.join(root, file))
                
                if result_files:
                    # 找到最新的文件
                    result_files.sort(key=os.path.getmtime, reverse=True)
                    latest_file = result_files[0]
                    QMessageBox.information(self, "检测完成", 
                                          f"摄像头检测已停止，最新结果已保存到: {latest_file}")
                else:
                    QMessageBox.information(self, "检测停止", "摄像头检测已停止")


class MainWindow(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        self.model_manager = ModelManager()
        self.init_ui()
        self.init_directories()
    
    def init_ui(self):
        self.setWindowTitle("YOLOv5 目标检测系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 设置应用程序样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F5F5;
            }
            QTabWidget::pane {
                border: 1px solid #CCCCCC;
                background-color: white;
                border-radius: 8px;
            }
            QTabWidget::tab-bar {
                left: 5px;
            }
            QTabBar::tab {
                background-color: #E9ECEF;
                color: #495057;
                padding: 12px 24px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-family: "Microsoft YaHei UI", "Segoe UI", "Arial", sans-serif;
                font-size: 14px;
                font-weight: 600;
                min-height: 20px;
                min-width: 140px;
            }
            QTabBar::tab:selected {
                background-color: #4A90E2;
                color: white;
                font-family: "Microsoft YaHei UI", "Segoe UI", "Arial", sans-serif;
                font-size: 14px;
                font-weight: 600;
                min-width: 140px;
                padding: 12px 24px;
            }
            QTabBar::tab:hover {
                background-color: #6BA6F0;
                color: white;
                font-family: "Microsoft YaHei UI", "Segoe UI", "Arial", sans-serif;
                font-size: 14px;
                font-weight: 600;
                min-width: 140px;
                padding: 12px 24px;
            }
        """)
        
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建选项卡
        tab_widget = QTabWidget()
        
        # 添加三个选项卡
        self.image_tab = ImageDetectionTab(self.model_manager)
        self.video_tab = VideoDetectionTab(self.model_manager)
        self.camera_tab = CameraDetectionTab(self.model_manager)
        
        tab_widget.addTab(self.image_tab, "🖼 图像检测")
        tab_widget.addTab(self.video_tab, "🎬 视频检测")
        tab_widget.addTab(self.camera_tab, "📹 摄像头检测")
        
        # 布局
        layout = QVBoxLayout()
        layout.addWidget(tab_widget)
        central_widget.setLayout(layout)
        
        # 设置窗口图标
        if os.path.exists("resource/UI.svg"):
            self.setWindowIcon(QIcon("resource/UI.svg"))
    
    def init_directories(self):
        """初始化必要的目录"""
        directories = ["result", "resource"]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)


class ImageDetectionThread(QThread):
    """图像检测线程"""
    finished = pyqtSignal(str)
    
    def __init__(self, image_path, model_path):
        super().__init__()
        self.image_path = image_path
        self.model_path = model_path
    
    def run(self):
        try:
            # 确保result目录存在
            result_dir = "result"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            
            # 调用检测函数（不需要显示标签）
            detect.run(
                source=self.image_path,
                weights=self.model_path,
                save_img=True,
                project=result_dir,
                show_label=None,
                view_img=False
            )
            
            # 查找生成的结果文件
            result_files = []
            for root, dirs, files in os.walk(result_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        result_files.append(os.path.join(root, file))
            
            if result_files:
                # 返回最新的结果文件
                result_files.sort(key=os.path.getmtime, reverse=True)
                self.finished.emit(result_files[0])
            else:
                self.finished.emit("")
                
        except Exception as e:
            print(f"检测错误: {e}")
            self.finished.emit("")


class VideoDetectionThread(QThread):
    """视频检测线程"""
    finished = pyqtSignal(str)
    
    def __init__(self, video_path, model_path):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
    
    def run(self):
        try:
            # 确保result目录存在
            result_dir = "result"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            
            # 调用检测函数（不需要显示标签）
            detect.run(
                source=self.video_path,
                weights=self.model_path,
                save_img=True,
                project=result_dir,
                show_label=None,
                view_img=False
            )
            
            # 查找生成的结果文件
            result_files = []
            for root, dirs, files in os.walk(result_dir):
                for file in files:
                    if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        result_files.append(os.path.join(root, file))
            
            if result_files:
                # 返回最新的结果文件
                result_files.sort(key=os.path.getmtime, reverse=True)
                self.finished.emit(result_files[0])
            else:
                self.finished.emit("")
                
        except Exception as e:
            print(f"检测错误: {e}")
            self.finished.emit("")


class CameraDetectionThread(QThread):
    """摄像头检测线程"""
    def __init__(self, model_path, display_label):
        super().__init__()
        self.model_path = model_path
        self.display_label = display_label
    
    def run(self):
        try:
            # 确保result目录存在
            result_dir = "result"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            
            detect.run(
                source=0,  # 摄像头
                weights=self.model_path,
                show_label=self.display_label,
                save_img=True,  # 启用保存图像
                project=result_dir,  # 指定保存目录
                use_camera=True
            )
        except Exception as e:
            print(f"摄像头检测错误: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 设置应用程序属性
    app.setApplicationName("YOLOv5检测系统")
    app.setApplicationVersion("1.0")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())
