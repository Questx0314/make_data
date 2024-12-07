import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import shutil
from core.data_processor import DataProcessor
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QMessageBox

# 添加SAM2包的路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sam2_path = os.path.join(project_root, 'sam2_repo')
sys.path.append(sam2_path)

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class ImageProcessor:
    MODEL_CONFIGS = {
        'Base': {
            'checkpoint': 'sam2.1_hiera_base.pt',
            'config': 'sam2.1_hiera_b.yaml'
        },
        'Large': {
            'checkpoint': 'sam2.1_hiera_large.pt',
            'config': 'sam2.1_hiera_l.yaml'
        },
        'Huge': {
            'checkpoint': 'sam2.1_hiera_huge.pt',
            'config': 'sam2.1_hiera_h.yaml'
        }
    }

    def __init__(self, model_size='Large', folders=None, progress_signal=None):
        self.device = self._initialize_device()
        self.sam2, self.mask_generator = self._initialize_model(model_size)
        self.progress_signal = progress_signal  # 用于进度更新的信号
        self.folders = folders  # 存储文件夹路径
        self.image_files = []  # 存储待处理的图片文件
        self.bounding_boxes = []  # 存储每张图片的边界框

    def _initialize_device(self):
        """初始化设备"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _initialize_model(self, model_size):
        """初始化SAM模型和掩码生成器"""
        model_config = self.MODEL_CONFIGS[model_size]
        sam2_checkpoint = os.path.join(
            project_root, 'models', 'checkpoints', 
            model_config['checkpoint']
        )
        model_cfg = os.path.join(
            project_root, 'models', 'configs', 'sam2.1',
            model_config['config']
        )
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=self.device, apply_postprocessing=False)
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=16,
            points_per_batch=64,
            pred_iou_thresh=0.9,
            stability_score_thresh=0.9,
            min_mask_region_area=9000,
            box_nms_thresh=0.7,
            use_m2m=True
        )
        return sam2, mask_generator

    def process_folder(self):
        """处理文件夹中的所有图片"""
        raw_folder = self.folders['raw']
        self.image_files = [f for f in os.listdir(raw_folder) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not self.image_files:
            raise ValueError("所选文件夹中没有图片文件！")

        # 创建处理线程
        self.processing_thread = ImageProcessingThread(self.image_files, self.mask_generator, self.folders)
        self.processing_thread.progress_signal.connect(self.handle_progress)
        self.processing_thread.start()

    def handle_progress(self, msg_type, message):
        """处理进度信息"""
        if msg_type == "error":
            QMessageBox.critical(self, "错误", message)
            self.enable_buttons()
        elif msg_type == "success":
            QMessageBox.information(self, "成功", message)
            # 这里可以选择不调用交互标注
            # self.start_interactive_annotation()  # 开始交互标注
        else:
            self.progress_signal.emit("info", message)  # 发送进度信息

    def start_interactive_annotation(self):
        """开始交互式标注"""
        # 注释掉交互标注的代码
        # for image_file, bounding_boxes in self.bounding_boxes:
        #     image_path = os.path.join(self.folders['raw'], image_file)
        #     image = cv2.imread(image_path)
        #     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #     # 更新当前处理图片信息
        #     if self.progress_signal:
        #         self.progress_signal.emit("info", f"当前处理图片: {image_file}")

        #     # 进行交互式标注
        #     self._interactive_annotation(image_path, image_rgb, bounding_boxes,
        #                                 self.folders['images'], self.folders['labels'], 
        #                                 self.folders['test'], self.folders['raw'])

        # 所有图片标注完成，开始数据增强
        self.start_data_augmentation(self.folders['images'], self.folders['labels'])

    def _interactive_annotation(self, image_path, image_rgb, bounding_boxes,
                              images_dir, labels_dir, test_dir, raw_folder):
        """交互式标注单张图片"""
        # 初始化变量
        selected_boxes = []
        preview_mode = False
        start_point = None
        end_point = None
        dragging = False
        drag_threshold = 5

        def draw_bounding_boxes():
            ax.clear()
            ax.imshow(image_rgb)

            # 显示提示文本
            if preview_mode:
                ax.text(0, -10, 'Preview mode: Press F to confirm, R to reselect.', 
                       fontsize=12, color='blue')
            else:
                ax.text(0, -10, 
                       'Drag to select/cancel boxes. Press F to preview. Press D to discard.',
                       fontsize=12, color='blue')

            # 绘制边界框
            boxes_to_draw = selected_boxes if preview_mode else bounding_boxes
            for box in boxes_to_draw:
                color = 'g' if box in selected_boxes else 'r'
                x1, y1, x2, y2 = box
                rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)

            # 绘制选择框
            if dragging and start_point and end_point:
                sx, sy = start_point
                ex, ey = end_point
                drag_color = 'b' if ex > sx else 'orange'
                rect = Rectangle((sx, sy), ex-sx, ey-sy,
                               linewidth=2, edgecolor=drag_color, facecolor='none')
                ax.add_patch(rect)

            fig.canvas.draw()

        def is_box_selected(box, sx, sy, ex, ey):
            x1, y1, x2, y2 = box
            if ex > sx:  # 从左到右选择：完全包含
                return sx <= x1 and sy <= y1 and ex >= x2 and ey >= y2
            else:  # 从右到左选择：相交
                def is_point_in_box(px, py, x1, y1, x2, y2):
                    return x1 <= px <= x2 and y1 <= py <= y2
                points = [(sx,sy), (ex,ey), (sx,ey), (ex,sy)]
                points_in_box = sum(is_point_in_box(px,py,x1,y1,x2,y2) for px,py in points)
                return points_in_box in [1, 2]

        def on_click(event):
            nonlocal start_point, end_point, dragging
            if preview_mode or not event.inaxes:
                return
            start_point = (event.xdata, event.ydata)
            dragging = True

        def on_release(event):
            nonlocal selected_boxes, start_point, end_point, dragging
            if preview_mode or not dragging or not event.inaxes:
                return

            end_point = (event.xdata, event.ydata)
            dragging = False

            if start_point and end_point:
                if np.sqrt((start_point[0] - end_point[0])**2 + 
                          (start_point[1] - end_point[1])**2) > drag_threshold:
                    sx, sy = start_point
                    ex, ey = end_point
                    for box in bounding_boxes:
                        if is_box_selected(box, sx, sy, ex, ey):
                            if box in selected_boxes:
                                selected_boxes.remove(box)
                            else:
                                selected_boxes.append(box)

            draw_bounding_boxes()

        def on_motion(event):
            nonlocal end_point
            if dragging and event.inaxes:
                end_point = (event.xdata, event.ydata)
                draw_bounding_boxes()

        def on_key(event):
            nonlocal preview_mode
            if event.key == 'f':
                if preview_mode:
                    # 保存结果
                    if selected_boxes:
                        image_output_path = os.path.join(images_dir, os.path.basename(image_path))
                        cv2.imwrite(image_output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
                        
                        label_output_path = os.path.join(labels_dir, 
                                                       os.path.splitext(os.path.basename(image_path))[0] + ".txt")
                        self._save_yolo_labels(label_output_path, selected_boxes, image_rgb.shape[:2])
                    plt.close()
                else:
                    preview_mode = True
                    draw_bounding_boxes()
            elif event.key == 'r' and preview_mode:
                preview_mode = False
                draw_bounding_boxes()
            elif event.key == 'd':
                # 移动到测试集
                test_path = os.path.join(test_dir, os.path.basename(image_path))
                cv2.imwrite(test_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
                plt.close()

        # 创建图像窗口
        fig, ax = plt.subplots(figsize=(15, 10))
        draw_bounding_boxes()

        # 绑定事件
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show()

    def _save_yolo_labels(self, output_file, bounding_boxes, image_shape, class_id=0):
        """保存YOLO格式的标签"""
        h, w = image_shape[:2]
        with open(output_file, "w") as f:
            for box in bounding_boxes:
                x1, y1, x2, y2 = box
                x_center = (x1 + x2) / 2.0 / w
                y_center = (y1 + y2) / 2.0 / h
                box_width = (x2 - x1) / w
                box_height = (y2 - y1) / h
                f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")

    def start_data_augmentation(self, images_dir, labels_dir):
        """开始数据增强"""
        from core.data_processor import DataProcessor
        data_processor = DataProcessor(self.folders, self.progress_signal)
        if data_processor.process_data():
            # 删除process文件夹
            shutil.rmtree(self.folders['process'])
            # 删除临时文件夹
            # shutil.rmtree(self.folders['temp'])
            self.progress_signal.emit("success", "处理完成！")

class ImageProcessingThread(QThread):
    progress_signal = pyqtSignal(str, str)

    def __init__(self, image_files, mask_generator, folders):
        super().__init__()
        self.image_files = image_files
        self.mask_generator = mask_generator
        self.folders = folders
        self.bounding_boxes = []

    def run(self):
        total_images = len(self.image_files)

        for idx, image_file in enumerate(self.image_files):
            image_path = os.path.join(self.folders['raw'], image_file)
            print(f"处理图片: {image_path}")

            # 读取并处理图片
            image = cv2.imread(image_path)
            image = cv2.resize(image, (1920, 1080))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 生成掩码和边界框
            masks = self.mask_generator.generate(image_rgb)
            bounding_boxes = []
            for mask in masks:
                mask_np = mask["segmentation"]
                x, y, w, h = cv2.boundingRect(mask_np.astype(np.uint8))
                bounding_boxes.append((x, y, x + w, y + h))

            # 将结果存储
            self.bounding_boxes.append((image_file, bounding_boxes))

            # 更新进度
            progress_percentage = int((idx + 1) / total_images * 100)
            self.progress_signal.emit("info", f"已处理 {idx + 1}/{total_images} 张图片 ({progress_percentage}%)")

        # 所有图片处理完成
        self.progress_signal.emit("success", "所有图片处理完成！")