import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from queue import Queue
import threading 
import sys
import shutil
from core.data_processor import DataProcessor


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

    def __init__(self, model_size='Large', progress_signal=None):
        # 初始化设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        # 获取模型配置
        model_config = self.MODEL_CONFIGS[model_size]
        
        # 构建模型路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.sam2_checkpoint = os.path.join(
            project_root, 'models', 'checkpoints', 
            model_config['checkpoint']
        )
        self.model_cfg = os.path.join(
            project_root, 'models', 'configs', 'sam2.1',
            model_config['config']
        )
        
        # 初始化SAM2模型
        self.sam2 = build_sam2(self.model_cfg, self.sam2_checkpoint, 
                              device=self.device, apply_postprocessing=False)
        
        # 初始化mask生成器
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2,
            points_per_side=16,
            points_per_batch=64,
            pred_iou_thresh=0.9,
            stability_score_thresh=0.9,
            min_mask_region_area=9000,
            box_nms_thresh=0.7,
            use_m2m=True
        )

        self.progress_signal = progress_signal  # 用于进度更新的信号
        # 初始化信号量
        self.semaphore = threading.Semaphore(0)
        self.current_image_index = 0  # 当前处理的图片索引
        self.image_files = []  # 存储待处理的图片文件
        self.folders = None

    def process_folder(self, folders):
        """处理文件夹中的所有图片"""
        raw_folder = folders['raw']
        images_dir = folders['images']
        labels_dir = folders['labels']
        test_dir = folders['test']
        self.folders = folders

        self.image_files = [f for f in os.listdir(raw_folder) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not self.image_files:
            raise ValueError("所选文件夹中没有图片文件！")

        # 处理每张图片
        for image_file in self.image_files:
            image_path = os.path.join(raw_folder, image_file)
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

            # 进行交互式标注
            self._interactive_annotation(image_path, image_rgb, bounding_boxes,
                                        images_dir, labels_dir, test_dir, raw_folder)

            # 从raw文件夹中移除已处理的图片
            os.remove(image_path)

        # 所有图片标注完成，释放信号量，开始数据增强
        self.start_data_augmentation(images_dir, labels_dir)

    def process_single_image(self, raw_folder, images_dir, labels_dir, test_dir):
        """处理单张图片"""
        if self.current_image_index >= len(self.image_files):
            # 所有图片处理完成，释放信号量
            self.semaphore.release()
            return

        image_file = self.image_files[self.current_image_index]
        image_path = os.path.join(raw_folder, image_file)
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

        # 进行交互式标注
        self._interactive_annotation(image_path, image_rgb, bounding_boxes,
                                    images_dir, labels_dir, test_dir, raw_folder)

        # 从raw文件夹中移除已处理的图片
        os.remove(image_path)

        # 更新当前图片索引并处理下一张
        self.current_image_index += 1
        self.process_single_image(raw_folder, images_dir, labels_dir, test_dir)

    def start_data_augmentation(self, images_dir, labels_dir):
        """开始数据增强"""
        from core.data_processor import DataProcessor
        data_processor = DataProcessor(self.folders, self.progress_signal)
        if data_processor.process_data():
            # 删除process文件夹
            shutil.rmtree(self.folders['process'])
            self.progress_signal.emit("success", "处理完成！")

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
                    
                    # 释放信号量，表示当前图片处理完成
                    self.semaphore.release()  # 释放信号量，表示当前图片处理完成
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
                # 释放信号量，表示当前图片处理完成
                self.semaphore.release()

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
