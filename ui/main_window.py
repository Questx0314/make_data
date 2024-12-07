from PyQt6.QtWidgets import (QMainWindow, QWidget, QPushButton, QTextEdit,
                             QHBoxLayout, QVBoxLayout, QFileDialog, QLabel,
                             QMessageBox, QGroupBox, QGridLayout, QSpinBox,
                             QDoubleSpinBox, QComboBox, QProgressBar)
from PyQt6.QtCore import Qt, pyqtSignal
import os
import torch
from core.image_processor import ImageProcessor
from core.trainer import TrainThread
from core.validator import ValidatorThread
import shutil
import yaml
import threading

class MainWindow(QMainWindow):
    progress_signal = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_variables()
        self.connect_signals()

    def init_ui(self):
        """初始化UI组件"""
        self.setWindowTitle("数据集生成工具")
        self.setGeometry(100, 100, 800, 600)
        
        # 创建中心部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 创建左侧布局
        left_widget = self.create_left_panel()
        
        # 创建右侧显示区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 添加状态显示标签
        self.status_label = QLabel("就绪")
        right_layout.addWidget(self.status_label)

        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        right_layout.addWidget(self.progress_bar)

        # 添加当前处理图片信息的文本框
        self.current_image_label = QLabel("当前处理图片: ")
        right_layout.addWidget(self.current_image_label)

        # 添加到主布局
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 2)

        # 添加状态栏
        self.statusBar = self.statusBar()
        self.progress_label = QLabel()
        self.statusBar.addWidget(self.progress_label)

        # 初始化控件状态
        self.set_controls_enabled(1)  # 只启用选择文件夹的控件

    def create_left_panel(self):
        """创建左侧面板"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # 文件选择区域
        file_select_widget = QWidget()
        file_select_layout = QHBoxLayout(file_select_widget)
        
        self.select_folder_btn = QPushButton("选择文件夹")
        self.select_folder_btn.setMaximumWidth(100)
        
        self.path_text = QTextEdit()
        self.path_text.setReadOnly(True)
        self.path_text.setMaximumHeight(30)
        
        file_select_layout.addWidget(self.select_folder_btn)
        file_select_layout.addWidget(self.path_text)

        # 模型设置区域
        model_group = QGroupBox("模型")
        model_layout = QGridLayout()
        
        model_layout.addWidget(QLabel("模型大小:"), 0, 0)
        self.model_size_combo = QComboBox()
        self.model_size_combo.addItems(['Base', 'Large', 'Huge'])
        self.model_size_combo.setCurrentText('Large')
        model_layout.addWidget(self.model_size_combo, 0, 1)
        
        model_group.setLayout(model_layout)

        # 开始标定按钮
        self.start_btn = QPushButton("开始标定")
        self.start_btn.setMinimumHeight(50)

        # 训练设置区域
        train_group = QGroupBox("训练设置")
        train_layout = QGridLayout()
        
        # Epochs设置
        train_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(300)
        train_layout.addWidget(self.epochs_spin, 0, 1)
        
        # Image Size设置
        train_layout.addWidget(QLabel("Image Size:"), 1, 0)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 1280)
        self.imgsz_spin.setValue(640)
        train_layout.addWidget(self.imgsz_spin, 1, 1)
        
        # Confidence设置
        train_layout.addWidget(QLabel("Confidence:"), 2, 0)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 1.0)
        self.conf_spin.setValue(0.5)
        self.conf_spin.setSingleStep(0.1)
        train_layout.addWidget(self.conf_spin, 2, 1)
        
        # 选择设备
        train_layout.addWidget(QLabel("Device:"), 3, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(['CPU', 'GPU'])
        if torch.cuda.is_available():
            self.device_combo.setCurrentText('GPU')
        else:
            self.device_combo.setCurrentText('CPU')
        train_layout.addWidget(self.device_combo, 3, 1)
        
        train_group.setLayout(train_layout)

        # 开始训练按钮
        self.train_btn = QPushButton("开始训练")
        self.train_btn.setMinimumHeight(40)

        # 添加所有组件到左侧布局
        left_layout.addWidget(file_select_widget)
        left_layout.addWidget(model_group)
        left_layout.addWidget(self.start_btn)
        left_layout.addWidget(train_group)
        left_layout.addWidget(self.train_btn)
        left_layout.addStretch()

        return left_widget

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
                # 检查jpg图片
                image_files = [f for f in os.listdir(folder_path) 
                               if f.lower().endswith('.jpg')]
                
                if not image_files:
                    QMessageBox.warning(self, "警告", "所选文件夹中没有jpg图片！")
                    return
                
                # 更新路径显示
                self.path_text.setText(folder_path)
                self.folders = {'base': folder_path}
                
                # 启用模型设置区域的控件
                self.set_controls_enabled(2)  # 启用模型设置区域的控件
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"检查文件夹失败: {str(e)}")

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
            
            # 创建处理器实例
            model_size = self.model_size_combo.currentText()
            self.processor = ImageProcessor(model_size=model_size, folders=self.folders, progress_signal=self.progress_signal)
            
            # 禁用按钮
            self.select_folder_btn.setEnabled(False)
            self.start_btn.setEnabled(False)
            self.model_size_combo.setEnabled(False)
            
            # 开始处理文件夹里的所有图片
            self.processor.process_folder()  # 直接调用处理方法
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理失败: {str(e)}")
        finally:
            self.enable_buttons()

    def process_images(self):
        """后台处理图片"""
        self.processor.process_folder(self.folders)  # 传递文件夹字典
        self.progress_signal.emit("success", "所有图片处理完成！")

    def handle_progress(self, msg_type, message):
        """处理进度信息"""
        if msg_type == "error":
            QMessageBox.critical(self, "错误", message)
            self.enable_buttons()
        elif msg_type == "success":
            QMessageBox.information(self, "成功", message)
            self.enable_buttons()
        else:
            self.progress_label.setText(message)  # 更新进度信息
            # 解析进度信息以更新进度条
            if "已处理" in message:
                progress_percentage = int(message.split(' ')[-1][:-1])  # 提取百分比
                self.progress_bar.setValue(progress_percentage)  # 更新进度条
                self.current_image_label.setText(message)  # 更新当前处理图片信息
            self.statusBar.repaint()

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

            epochs = self.epochs_spin.value()
            imgsz = self.imgsz_spin.value()
            device = self.device_combo.currentText()
            conf = self.conf_spin.value()

            # 创建练线程
            self.train_thread = TrainThread(
                yaml_path=yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                device=device,
                conf=conf
            )
            self.train_thread.progress_signal.connect(self.handle_progress)
            self.train_thread.finished_signal.connect(self.start_validation)

            # 禁用按钮
            # self.train_btn.setEnabled(False)
            # self.start_btn.setEnabled(False)
            # self.set_controls_enabled(3)

            # 开始训练
            self.train_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动训练失败: {str(e)}")
            self.enable_buttons()

    def start_validation(self, model_path):
        """开始验证"""
        try:
            # 创建验结果保存目录
            save_folder = os.path.join(self.folders['base'], 'validation_results')
            
            # 创建验证程
            self.validator_thread = ValidatorThread(
                model_path=model_path,
                test_folder=self.folders['test'],
                save_folder=save_folder,
                conf=self.conf_spin.value()
            )
            
            self.validator_thread.progress_signal.connect(self.handle_progress)
            self.validator_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动验证失败: {str(e)}")
            self.enable_buttons()

    def enable_buttons(self):
        """启用所有按钮"""
        self.select_folder_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.model_size_combo.setEnabled(True)
        self.train_btn.setEnabled(True)

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
            self.conf_spin.setEnabled(False)
            self.imgsz_spin.setEnabled(False)
            self.epochs_spin.setEnabled(False)
            self.device_combo.setEnabled(False)
        elif step == 2:  # 标定
            self.select_folder_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.model_size_combo.setEnabled(True)
            self.train_btn.setEnabled(False)
            self.conf_spin.setEnabled(False)
            self.imgsz_spin.setEnabled(False)
            self.epochs_spin.setEnabled(False)
            self.device_combo.setEnabled(False)
        elif step == 3:  # 训练
            self.select_folder_btn.setEnabled(False)
            self.start_btn.setEnabled(False)
            self.model_size_combo.setEnabled(False)
            self.train_btn.setEnabled(True)
            self.conf_spin.setEnabled(True)
            self.imgsz_spin.setEnabled(True)
            self.epochs_spin.setEnabled(True)
            self.device_combo.setEnabled(False)