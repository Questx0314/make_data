from PyQt6.QtCore import QThread, pyqtSignal
import os
import cv2
import random
import shutil
from albumentations import (Compose, HorizontalFlip, VerticalFlip, 
                          RandomBrightnessContrast, Rotate, RandomScale, 
                          ShiftScaleRotate)
from albumentations.core.composition import BboxParams
import yaml

class DataProcessor:
    def __init__(self, folders, progress_signal=None):
        self.folders = folders
        self.transform = self._setup_transform()
        self.progress_signal = progress_signal  # 添加进度信号

    def process_data(self):
        """处理数据：增强数据并生成yaml文件"""
        try:
            # 创建train和val目录
            for subset in ['train', 'val']:
                os.makedirs(os.path.join(self.folders['images'], subset), exist_ok=True)
                os.makedirs(os.path.join(self.folders['labels'], subset), exist_ok=True)

            # 数据增强
            if not self.augment_data():
                raise Exception("数据增强失败")
            
            # 生成yaml文件
            self.generate_yaml()
            
            return True
        except Exception as e:
            if self.progress_signal:
                self.progress_signal.emit("error", f"数据处理失败: {str(e)}")
            return False

    def generate_yaml(self):
        """生成YOLO训练所需的YAML配置文件"""
        data_yaml = {
            'path': self.folders['base'],  # 数据集根目录
            'train': "images/train",       # 训练集路径
            'val': "images/val",           # 验证集路径
            'names': {0: "targets"}        # 类别名称
        }

        output_file = os.path.join(self.folders['base'], 'data.yaml')
        with open(output_file, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

    def _setup_transform(self):
        """设置数据增强转换"""
        return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.2),
            RandomBrightnessContrast(p=0.2),
            Rotate(limit=15, p=0.5),
            RandomScale(scale_limit=0.2, p=0.5),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5)
        ], bbox_params=BboxParams(format='yolo', label_fields=['class_labels']))

    def augment_data(self, num_augmentations=5):
        """数据增强"""
        try:
            if self.progress_signal:
                self.progress_signal.emit("info", "开始数据增强...")

            # 从images和labels文件夹获取原始数据
            original_images = [f for f in os.listdir(self.folders['images']) 
                             if f.endswith('.jpg')]
            
            if not original_images:
                raise ValueError("没有找到原始图片，请先完成图片标注")
            
            if self.progress_signal:
                self.progress_signal.emit("info", f"找到 {len(original_images)} 张原始图片")

            # 对每张原始图片进行增强
            for idx, img_file in enumerate(original_images):
                if self.progress_signal:
                    self.progress_signal.emit("info", f"正在处理第 {idx+1}/{len(original_images)} 张图片")

                # 读取图片和标签
                image_path = os.path.join(self.folders['images'], img_file)
                label_path = os.path.join(self.folders['labels'], 
                                        img_file.replace('.jpg', '.txt'))
                
                if not os.path.exists(label_path):
                    print(f"警告：找不到标签文件 {label_path}")
                    continue

                # 读取图片和标签
                image = cv2.imread(image_path)
                if image is None:
                    print(f"警告：无法读取图片 {image_path}")
                    continue
                
                bboxes, class_labels = self._read_labels(label_path)

                # 先复制原始文件到process文件夹
                shutil.copy2(image_path, os.path.join(self.folders['process'], img_file))
                shutil.copy2(label_path, os.path.join(self.folders['process'], 
                                                    img_file.replace('.jpg', '.txt')))

                # 执行数据增强
                for i in range(num_augmentations):
                    try:
                        augmented = self.transform(
                            image=image,
                            bboxes=bboxes,
                            class_labels=class_labels
                        )
                        
                        # 生成增强后的文件名
                        aug_name = f"{os.path.splitext(img_file)[0]}_aug_{i}"
                        aug_image_path = os.path.join(self.folders['process'], 
                                                    f"{aug_name}.jpg")
                        aug_label_path = os.path.join(self.folders['process'], 
                                                    f"{aug_name}.txt")
                        
                        # 保存增强后的图片和标签
                        cv2.imwrite(aug_image_path, augmented['image'])
                        self._save_labels(aug_label_path, 
                                        augmented['bboxes'],
                                        augmented['class_labels'])
                        
                    except Exception as e:
                        print(f"增强图片 {img_file} 第 {i+1} 次失败: {str(e)}")
                        continue

            if self.progress_signal:
                self.progress_signal.emit("info", "数据增强完成，开始分割数据集...")

            # 分割数据集并移动到对应文件夹
            self.split_train_val(train_ratio=0.8)
            
            return True
            
        except Exception as e:
            if self.progress_signal:
                self.progress_signal.emit("error", f"数据增强失败: {str(e)}")
            print(f"数据增强失败: {str(e)}")  # 打印详细错误信息
            return False

    def _read_labels(self, label_path):
        """读取标签文件"""
        bboxes = []
        class_labels = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_labels.append(int(parts[0]))
                bboxes.append([float(p) for p in parts[1:]])
                
        return bboxes, class_labels

    def _save_labels(self, label_path, bboxes, class_labels):
        """保存标签文件"""
        with open(label_path, 'w') as f:
            for bbox, cls in zip(bboxes, class_labels):
                f.write(f"{cls} " + " ".join(map(str, bbox)) + "\n")

    def split_train_val(self, train_ratio=0.8):
        """分割训练集和验证集"""
        if self.progress_signal:
            self.progress_signal.emit("info", "开始创建训练和验证集目录...")
            
        # 创建训练和验证目录
        for subset in ['train', 'val']:
            os.makedirs(os.path.join(self.folders['images'], subset), exist_ok=True)
            os.makedirs(os.path.join(self.folders['labels'], subset), exist_ok=True)

        # 获取process文件夹中的所有文件
        files = [(f, f.replace('.jpg', '.txt')) 
                 for f in os.listdir(self.folders['process']) 
                 if f.endswith('.jpg')]
        
        if self.progress_signal:
            self.progress_signal.emit("info", f"找到 {len(files)} 个文件待分配")
        
        # 随机打乱文件列表
        random.shuffle(files)
        
        # 按8:2的比例分割
        split_idx = int(len(files) * train_ratio)
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        if self.progress_signal:
            self.progress_signal.emit(
                "info", 
                f"分配比例: 训练集 {len(train_files)} 个, 验证集 {len(val_files)} 个"
            )
        
        # 移动文件到对应目录
        for files, subset in [(train_files, 'train'), (val_files, 'val')]:
            if self.progress_signal:
                self.progress_signal.emit("info", f"正在移动文件到{subset}目录...")
                
            moved = 0
            for img_file, label_file in files:
                # 移动图片和标签到对应目录
                shutil.move(
                    os.path.join(self.folders['process'], img_file),
                    os.path.join(self.folders['images'], subset, img_file)
                )
                shutil.move(
                    os.path.join(self.folders['process'], label_file),
                    os.path.join(self.folders['labels'], subset, label_file)
                )
                moved += 1
                
                if self.progress_signal and moved % 10 == 0:  # 每10个文件更新一次进度
                    self.progress_signal.emit(
                        "info", 
                        f"{subset}目录: 已移动 {moved}/{len(files)} 个文件"
                    )
        
        if self.progress_signal:
            self.progress_signal.emit("info", "数据集分割完成！")

    def _rename_files(self, train_dirs):
        """重命名文件"""
        for subset in ['train', 'val']:
            images_dir = os.path.join(self.folders['process'], subset)
            labels_dir = images_dir  # 标签和图片在同一目录
            
            image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
            for idx, image_file in enumerate(image_files):
                base_name = os.path.splitext(image_file)[0]
                label_file = f"{base_name}.txt"
                
                new_image_name = f"image_{idx:04d}.jpg"
                new_label_name = f"image_{idx:04d}.txt"
                
                os.rename(os.path.join(images_dir, image_file),
                         os.path.join(images_dir, new_image_name))
                os.rename(os.path.join(labels_dir, label_file),
                         os.path.join(labels_dir, new_label_name))
