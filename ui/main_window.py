from PyQt6.QtWidgets import (QMainWindow, QWidget, QPushButton, QTextEdit, QLabel,
                             QMessageBox, QFileDialog, QComboBox, QProgressBar, QLineEdit)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QIntValidator, QDoubleValidator
import os
import torch
import shutil

class MainWindow(QMainWindow):
    progress_signal = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_variables()
        self.connect_signals()
        self.set_controls_enabled(1)

    def init_ui(self):
        """初始化UI组件"""
        self.setWindowTitle("数据集生成工具")
        self.setGeometry(100, 100, 800, 600)

        # 创建中心部件
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # 文件选择区域
        self.select_folder_btn = QPushButton("选择文件夹", self)
        self.select_folder_btn.setGeometry(50, 50, 100, 30)  # 设置位置和大小
        self.path_text = QTextEdit(self)
        self.path_text.setGeometry(160, 50, 300, 30)  # 设置位置和大小
        self.path_text.setReadOnly(True)

        # 模型设置区域
        QLabel("模型大小:", self).setGeometry(50, 100, 100, 30)  # 标签位置
        self.model_size_combo = QComboBox(self)
        self.model_size_combo.setGeometry(160, 100, 100, 30)  # 设置位置和大小
        self.model_size_combo.addItems(['Base', 'Large', 'Huge'])
        self.model_size_combo.setCurrentText('Large')

        # 开始标定按钮
        self.start_btn = QPushButton("开始标定", self)
        self.start_btn.setGeometry(50, 150, 100, 30)  # 设置位置和大小

        # 训练设置区域
        QLabel("训练设置", self).setGeometry(50, 200, 100, 30)  # 标签位置

        # Epochs设置
        QLabel("Epochs:", self).setGeometry(50, 230, 100, 30)  # 标签位置
        self.epochs_input = QLineEdit(self)
        self.epochs_input.setGeometry(160, 230, 100, 30)  # 设置位置和大小
        self.epochs_input.setPlaceholderText("输入Epochs")
        self.epochs_input.setValidator(QIntValidator(1, 1000, self))  # 只接受1到1000的整数

        # Image Size设置
        QLabel("Image Size:", self).setGeometry(50, 270, 100, 30)  # 标签位置
        self.imgsz_input = QLineEdit(self)
        self.imgsz_input.setGeometry(160, 270, 100, 30)  # 设置位置和大小
        self.imgsz_input.setPlaceholderText("输入Image Size")
        self.imgsz_input.setValidator(QIntValidator(32, 1280, self))  # 只接受32到1280的整数

        # Confidence设置
        QLabel("Confidence:", self).setGeometry(50, 310, 100, 30)  # 标签位置
        self.conf_input = QLineEdit(self)
        self.conf_input.setGeometry(160, 310, 100, 30)  # 设置位置和大小
        self.conf_input.setPlaceholderText("输入Confidence")
        self.conf_input.setValidator(QDoubleValidator(0.1, 1.0, 2, self))  # 只接受0.1到1.0的浮点数

        # 选择设备
        QLabel("Device:", self).setGeometry(50, 350, 100, 30)  # 标签位置
        self.device_combo = QComboBox(self)
        self.device_combo.setGeometry(160, 350, 100, 30)  # 设置位置和大小
        self.device_combo.addItems(['CPU', 'GPU'])
        if torch.cuda.is_available():
            self.device_combo.setCurrentText('GPU')
        else:
            self.device_combo.setCurrentText('CPU')

        # 开始训练按钮
        self.train_btn = QPushButton("开始训练", self)
        self.train_btn.setGeometry(50, 400, 100, 30)  # 设置位置和大小

        # 添加状态显示标签
        self.status_label = QLabel("就绪", self)
        self.status_label.setGeometry(50, 460, 200, 30)  # 设置位置和大小

        # 添加进度条
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(50, 490, 400, 30)  # 设置位置和大小
        self.progress_bar.setRange(0, 100)

        # 添加当前处理图片信息的文本框
        self.current_image_label = QLabel("当前处理图片: ", self)
        self.current_image_label.setGeometry(50, 530, 400, 30)  # 设置位置和大小

        # 添加状态栏
        self.statusBar = self.statusBar()
        self.progress_label = QLabel()
        self.statusBar.addWidget(self.progress_label)

        # 初始化控件状态
        self.set_controls_enabled(1)  # 只启用选择文件夹的控件

    def init_variables(self):
        """初始化变量"""
        self.processor = None
        self.train_thread = None
        self.validator_thread = None
        self.folders = None

    def connect_signals(self):
        """连接信号和槽"""
        self.select_folder_btn.clicked.connect(self.select_folder)
        self.start_btn.clicked.connect(self.start_processing)
        self.train_btn.clicked.connect(self.start_training)

    def select_folder(self):
        """选择文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder_path:
            try:
                # 检查PNG文件
                png_files = [f for f in os.listdir(folder_path) 
                             if f.lower().endswith('.jpg')]
                
                if not png_files:
                    QMessageBox.warning(self, "警告", "所选文件夹中没有jpg文件！")
                    return
                
                # 更新路径显示
                self.path_text.setText(folder_path)
                self.folders = {'base': folder_path}
                
                # 启用模型设置区域的控件
                self.set_controls_enabled(2)  # 启用模型设置区域的控件
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"选择文件夹失败: {str(e)}")

    def start_processing(self):
        """开始处理"""
        if not self.folders:
            QMessageBox.warning(self, "警告", "请先选择文件夹！")
            return

        try:
            # 创建文件夹结构
            for folder in ['raw', 'images', 'labels', 'test', 'process']:
                path = os.path.join(self.folders['base'], folder)
                os.makedirs(path, exist_ok=True)
                self.folders[folder] = path
                
            # 移动图片到raw文件夹
            image_files = [f for f in os.listdir(self.folders['base']) 
                          if f.lower().endswith('.jpg')]
            for image_file in image_files:
                src = os.path.join(self.folders['base'], image_file)
                dst_raw = os.path.join(self.folders['raw'], image_file)
                shutil.copy(src, dst_raw)  # 复制到raw文件夹
            

            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理失败: {str(e)}")
        finally:
            self.set_controls_enabled(3)

    def start_training(self):
        """开始训练"""
        if not self.folders:
            QMessageBox.warning(self, "警告", "请先处理数据！")
            return

        try:
            # 获取训练参数
            yaml_path = os.path.join(self.folders['base'], 'data.yaml')
            if not os.path.exists(yaml_path):
                QMessageBox.warning(self, "警告", "找不到data.yaml件，请先完成数据处理！")
                return

            epochs = int(self.epochs_input.text())  # 从文本框获取值
            imgsz = int(self.imgsz_input.text())  # 从文本框获取值
            device = self.device_combo.currentText()
            conf = float(self.conf_input.text())  # 从文本框获取值

        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动训练失败: {str(e)}")
            self.enable_buttons()

    def closeEvent(self, event):
        """关闭窗口时删除文件夹及其内容"""
        try:
            if self.folders:  # 确保文件夹存在
                for folder in ['raw', 'images', 'labels', 'test', 'process']:
                    folder_path = self.folders.get(folder)
                    if folder_path and os.path.exists(folder_path):
                        shutil.rmtree(folder_path)  # 删除文件夹及其内容
            event.accept()  # 确认关闭
        except Exception as e:
            QMessageBox.critical(self, "错误", f"删除文件夹时出错: {str(e)}")
            event.ignore()  # 阻止窗口关闭，直到错误被解决

    def set_controls_enabled(self, step):
        """根据步骤禁用或启用控件"""
        if step == 1:  # 选择文件夹
            self.select_folder_btn.setEnabled(True)
            self.start_btn.setEnabled(False)
            self.model_size_combo.setEnabled(False)
            self.train_btn.setEnabled(False)
            self.conf_input.setEnabled(False)
            self.imgsz_input.setEnabled(False)
            self.epochs_input.setEnabled(False)
            self.device_combo.setEnabled(False)
        elif step == 2:  # 标定
            self.select_folder_btn.setEnabled(False)
            self.start_btn.setEnabled(True)
            self.model_size_combo.setEnabled(True)
            self.train_btn.setEnabled(False)
            self.conf_input.setEnabled(False)
            self.imgsz_input.setEnabled(False)
            self.epochs_input.setEnabled(False)
            self.device_combo.setEnabled(False)
        elif step == 3:  # 训练
            self.select_folder_btn.setEnabled(False)
            self.start_btn.setEnabled(False)
            self.model_size_combo.setEnabled(False)
            self.train_btn.setEnabled(True)
            self.conf_input.setEnabled(True)
            self.imgsz_input.setEnabled(True)
            self.epochs_input.setEnabled(True)
            self.device_combo.setEnabled(False)