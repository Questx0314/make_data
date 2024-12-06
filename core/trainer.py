import os
import torch
import torchvision
from ultralytics import YOLO
from PyQt6.QtCore import QThread, pyqtSignal

class TrainThread(QThread):
    progress_signal = pyqtSignal(str, str)
    finished_signal = pyqtSignal(str)
    
    def __init__(self, yaml_path, epochs, imgsz, device, conf):
        super().__init__()
        self.yaml_path = yaml_path
        self.epochs = epochs
        self.imgsz = imgsz
        self.device = 0 if device == 'GPU' else 'cpu'
        self.conf = conf
        
    def run(self):
        try:
            self.progress_signal.emit("info", "正在初始化训练...")
            
            # 设置环境变量
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            
            # 打印版本信息
            self.progress_signal.emit("info", f"PyTorch version: {torch.__version__}")
            self.progress_signal.emit("info", f"Torchvision version: {torchvision.__version__}")
            self.progress_signal.emit("info", f"CUDA is available: {torch.cuda.is_available()}")
            
            # 初始化模型
            model = YOLO('yolov8n.pt')
            
            # 开始训练
            self.progress_signal.emit("info", "开始训练...")
            weights_dir = os.path.join(os.path.dirname(self.yaml_path), 'weights')
            model.train(
                data=self.yaml_path,
                epochs=self.epochs,
                imgsz=self.imgsz,
                device=self.device,
                pretrained=True,
                conf=self.conf,
                project=weights_dir,
                name='train'
            )
            
            # 更新最佳模型路径
            best_model_path = os.path.join(weights_dir, 'train', 'weights', 'best.pt')
            self.finished_signal.emit(best_model_path)
            
            self.progress_signal.emit("success", "训练完成！")
            
        except Exception as e:
            self.progress_signal.emit("error", f"训练失败: {str(e)}") 