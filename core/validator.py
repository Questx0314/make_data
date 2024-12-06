import os
import cv2
import torch
from ultralytics import YOLO
from PyQt6.QtCore import QThread, pyqtSignal
import datetime

class ValidatorThread(QThread):
    progress_signal = pyqtSignal(str, str)
    
    def __init__(self, model_path, test_folder, save_folder, conf=0.5):
        super().__init__()
        self.model_path = model_path
        self.test_folder = test_folder
        self.save_folder = save_folder
        self.conf = conf
        
    def run(self):
        try:
            self.progress_signal.emit("info", "加载模型...")
            model = YOLO(self.model_path)
            
            # 获取测试图片列表
            image_files = [f for f in os.listdir(self.test_folder) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if not image_files:
                self.progress_signal.emit("error", "测试文件夹中没有图片！")
                return
                
            # 为每次验证创建时间戳子目录
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(self.save_folder, timestamp)
            os.makedirs(save_dir, exist_ok=True)
            
            # 处理每张图片
            for img_file in image_files:
                self.progress_signal.emit("info", f"处理图片: {img_file}")
                
                # 读取图片
                img_path = os.path.join(self.test_folder, img_file)
                
                # 进行预测
                results = model(img_path, conf=self.conf)[0]
                
                # 在图片上绘制预测结果
                img = cv2.imread(img_path)
                for box in results.boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # 绘制边界框
                    cv2.rectangle(img, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                (0, 255, 0), 2)
                    
                    # 添加置信度标签
                    label = f"{conf:.2f}"
                    cv2.putText(img, label, 
                              (int(x1), int(y1)-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 2)
                
                # 保存结果
                save_path = os.path.join(save_dir, f"pred_{img_file}")
                cv2.imwrite(save_path, img)
                
            self.progress_signal.emit("success", "验证完成！")
            
        except Exception as e:
            self.progress_signal.emit("error", f"验证失败: {str(e)}") 