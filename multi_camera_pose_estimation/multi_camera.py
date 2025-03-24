# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import time
import torch
import threading
import queue
import multiprocessing as mp
from mmpose.apis.inferencers import MMPoseInferencer
from mmpose.apis.inference_tracking import _compute_iou  # 导入MMPose提供的IOU计算函数
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional

# 全局变量控制程序退出
running = True

# 使用上下文管理器进行CUDA异步处理
@contextmanager
def torch_inference_mode():
    """使用torch推理模式的上下文管理器，优化推理性能"""
    with torch.inference_mode(), torch.amp.autocast(device_type='cuda', enabled=True):
        yield

class FrameProcessor:
    """处理视频帧的类，支持异步操作"""
    
    def __init__(self, model_name: str, device: str, 
                 input_queue_size: int = 3, 
                 output_queue_size: int = 3,
                 model_config: Dict = None):
        """
        初始化帧处理器
        
        Args:
            model_name: 要使用的模型名称
            device: 使用的设备('cpu'或'cuda:0'等)
            input_queue_size: 输入队列大小
            output_queue_size: 输出队列大小
            model_config: 模型配置参数
        """
        self.device = device
        self.model_name = model_name
        self.input_queue = queue.Queue(maxsize=input_queue_size)
        self.output_queue = queue.Queue(maxsize=output_queue_size)
        
        # 默认推理配置
        self.call_args = {
            'show': False,  # 我们自己处理显示
            'draw_bbox': False,
            'radius': 5,
            'thickness': 2,
            'kpt_thr': 0.5,
            'bbox_thr': 0.3,
            'nms_thr': 0.65,  # 对于rtmpose，更高的NMS阈值通常更好
            'pose_based_nms': True,  # 启用基于姿态的NMS
            'max_num_bboxes': 15  # 增加检测的最大人数上限
        }

        
        # 如果提供了模型配置，则更新配置
        if model_config:
            self.call_args.update(model_config)
            
        # 延迟初始化模型，以便在线程中进行
        self.inferencer = None
        
        # 性能统计
        self.inference_times = []
        self.last_inference_time = 0
        
        # 启动处理线程
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()
        
    def _init_model(self):
        """初始化姿态估计模型"""
        try:
            self.inferencer = MMPoseInferencer(
                pose2d=self.model_name,
                device=self.device,
                scope='mmpose',
                show_progress=False
            )
            print(f"成功加载模型到 {self.device}")
            return True
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return False
            
    def _inference_worker(self):
        """推理线程的工作函数"""
        if not self._init_model():
            global running
            running = False
            return
            
        while running:
            try:
                # 尝试从队列获取帧，有1秒超时
                try:
                    frame_data = self.input_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                frame_id, frame = frame_data
                
                # 开始计时
                start_time = time.time()
                
                # 转换为RGB格式（MMPose期望RGB格式）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 使用torch推理模式提高性能
                with torch_inference_mode():
                    # 使用MMPose进行推理
                    results = list(self.inferencer(frame_rgb, **self.call_args))
                
                # 计算推理时间
                inference_time = time.time() - start_time
                self.last_inference_time = inference_time
                self.inference_times.append(inference_time)
                # 保持统计列表在合理大小
                if len(self.inference_times) > 30:
                    self.inference_times.pop(0)
                
                # 将结果放入输出队列
                self.output_queue.put((frame_id, frame, results, inference_time))
                self.input_queue.task_done()
                
            except Exception as e:
                print(f"推理过程中出错: {str(e)}")
                # 出错时也标记任务完成
                if 'frame_data' in locals():
                    self.input_queue.task_done()
    
    def add_frame(self, frame_id: int, frame: np.ndarray):
        """添加帧到处理队列
        
        Args:
            frame_id: 帧ID
            frame: 视频帧
        """
        try:
            self.input_queue.put((frame_id, frame), block=False)
            return True
        except queue.Full:
            # 如果队列已满，放弃这一帧
            return False
    
    def get_result(self, timeout: float = 0.1) -> Optional[Tuple]:
        """获取处理结果
        
        Args:
            timeout: 获取超时时间(秒)
            
        Returns:
            (frame_id, frame, results, inference_time)元组，或None(如果队列为空)
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def get_avg_inference_time(self) -> float:
        """获取平均推理时间"""
        if not self.inference_times:
            return 0
        return sum(self.inference_times) / len(self.inference_times)

class CameraCapture:
    """摄像头捕获类，运行在单独的线程中"""
    
    def __init__(self, camera_id: int = 0, queue_size: int = 3):
        """
        初始化摄像头捕获
        
        Args:
            camera_id: 摄像头ID
            queue_size: 帧队列大小
        """
        self.camera_id = camera_id
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.cap = None
        self.frame_count = 0
        
        # 启动捕获线程
        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.capture_thread.start()
        
    def _capture_worker(self):
        """捕获线程的工作函数"""
        # 创建摄像头对象
        self.cap = cv2.VideoCapture(self.camera_id)
        
        # 尝试设置更高的捕获分辨率和其他优化
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        if not self.cap.isOpened():
            print("无法打开摄像头")
            global running
            running = False
            return
            
        # 主捕获循环
        while running:
            ret, frame = self.cap.read()
            if not ret:
                print("无法获取视频帧")
                break
                
            self.frame_count += 1
            
            # 尝试将帧放入队列，如果队列已满则丢弃
            try:
                self.frame_queue.put((self.frame_count, frame), block=False)
            except queue.Full:
                pass  # 队列满时丢弃帧，保持实时性
                
        # 循环结束，释放摄像头
        if self.cap is not None:
            self.cap.release()
            
    def get_frame(self, timeout: float = 0.1) -> Optional[Tuple]:
        """获取一帧视频
        
        Args:
            timeout: 获取超时时间(秒)
            
        Returns:
            (frame_id, frame)元组，或None(如果队列为空)
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def release(self):
        """释放资源"""
        if self.cap is not None:
            self.cap.release()

def process_pose_results(frame: np.ndarray, results: List, call_args: Dict) -> np.ndarray:
    """处理姿态估计结果并绘制到帧上
    
    Args:
        frame: 原始视频帧
        results: 姿态估计结果
        call_args: 调用参数
        
    Returns:
        处理后的帧
    """
    display_frame = frame.copy()
    
    # 定义需要保留的关键点索引 (0-16: 身体关键点, 17-22: 足部关键点)
    kept_indices = list(range(0, 23))  # 0-22的索引
    
    # 定义骨架连接关系 (关键点索引对)，使用MMPose连接样式
    skeleton = [
        # 头部连接
        (0, 1), (0, 2),         # 鼻子到左/右眼
        (1, 3), (2, 4),         # 左/右眼到左/右耳
        # 上身躯干
        (0, 5), (0, 6), (5, 6), # 颈部三角形
        (5, 7), (7, 9),         # 左肩到左手腕
        (6, 8), (8, 10),        # 右肩到右手腕
        # 下半身
        (5, 11), (6, 12),       # 肩膀到髋部
        (11, 12),               # 左右髋部连接
        (11, 13), (13, 15),     # 左髋到左脚踝
        (12, 14), (14, 16),     # 右髋到右脚踝
        # 足部连接
        (15, 19), (15, 17),  
        (15, 18), (16, 22), 
        (16, 21), (16, 20)
    ]
    
    # 为骨架连接定义颜色
    link_colors = [
        (255, 0, 0),   # 鼻子到左眼 - 红色
        (255, 0, 0),   # 鼻子到右眼 - 红色
        (255, 0, 0),   # 左眼到左耳 - 红色
        (255, 0, 0),   # 右眼到右耳 - 红色
        (255, 165, 0), # 颈部连接 - 橙色
        (255, 165, 0), # 颈部连接 - 橙色
        (255, 165, 0), # 颈部连接 - 橙色
        (0, 255, 0),   # 左上肢 - 绿色
        (0, 255, 0),   # 左上肢 - 绿色
        (0, 0, 255),   # 右上肢 - 蓝色
        (0, 0, 255),   # 右上肢 - 蓝色
        (255, 255, 0), # 左肩到左髋 - 黄色
        (255, 0, 255), # 右肩到右髋 - 紫色
        (128, 128, 0), # 髋部连接 - 橄榄色
        (255, 255, 0), # 左下肢 - 黄色
        (255, 255, 0), # 左下肢 - 黄色
        (255, 0, 255), # 右下肢 - 紫色
        (255, 0, 255), # 右下肢 - 紫色
        (0, 255, 255), # 左脚 - 青色
        (0, 255, 255), # 左脚 - 青色
        (128, 0, 128), # 右脚 - 深紫色
        (128, 0, 128)  # 右脚 - 深紫色
    ]

    # 人员计数器
    person_count = 0

    # 遍历结果
    for result in results:
        pred_instances = result.get('predictions', [])
        
        # 如果有预测结果，在原始帧上绘制
        if pred_instances and len(pred_instances) > 0:
            # 处理每个预测实例
            for instance_list in pred_instances:
                # 检查是否为列表类型（多个人的情况）
                if isinstance(instance_list, list):
                    for instance in instance_list:
                        # 绘制单个人的姿态
                        process_single_person(display_frame, instance, person_count, kept_indices, 
                                             skeleton, link_colors, call_args)
                        person_count += 1
                # 检查是否为字典类型（单个人的情况）
                elif isinstance(instance_list, dict):
                    instance = instance_list
                    process_single_person(display_frame, instance, person_count, kept_indices, 
                                         skeleton, link_colors, call_args)
                    person_count += 1
    
    
    return display_frame

def process_single_person(display_frame, instance, person_idx, kept_indices, skeleton, link_colors, call_args):
    """处理单个人的姿态估计结果
    
    Args:
        display_frame: 显示帧
        instance: 单个人的姿态估计结果
        person_idx: 人员索引
        kept_indices: 保留的关键点索引
        skeleton: 骨架连接关系
        link_colors: 连接线颜色
        call_args: 调用参数
    """
    # 获取关键点和得分
    keypoints = instance.get('keypoints', None)
    keypoint_scores = instance.get('keypoint_scores', None)
    
    # 获取track_id（如果存在）或使用序号作为ID
    track_id = instance.get('track_id', person_idx)
    
    if keypoints is not None and keypoint_scores is not None:
        # 转换为numpy数组
        keypoints = np.array(keypoints)
        keypoint_scores = np.array(keypoint_scores)
        
        # 检查关键点数量是否足够
        if len(keypoints) >= max(kept_indices) + 1:
            # 计算新的关键点: 颈椎中点（鼻子和左右肩中点之间的点）
            neck_valid = (keypoint_scores[0] > call_args['kpt_thr'] and 
                         keypoint_scores[5] > call_args['kpt_thr'] and 
                         keypoint_scores[6] > call_args['kpt_thr'])
            
            if neck_valid:
                # 计算左右肩中点
                shoulder_mid_x = (keypoints[5][0] + keypoints[6][0]) / 2
                shoulder_mid_y = (keypoints[5][1] + keypoints[6][1]) / 2
                
                # 计算颈椎中点（鼻子和肩膀中点之间的某个位置，这里取1/3处）
                neck_vertebra_x = int(keypoints[0][0] * 0.3 + shoulder_mid_x * 0.7)
                neck_vertebra_y = int(keypoints[0][1] * 0.3 + shoulder_mid_y * 0.7)
                neck_vertebra_point = (neck_vertebra_x, neck_vertebra_y)
                
                # 绘制颈椎中点
                cv2.circle(display_frame, neck_vertebra_point, call_args['radius'], (0, 165, 255), -1)
            
            # 计算新的关键点: 髂前上棘连线中点（左右髋部的中点）
            hip_valid = (keypoint_scores[11] > call_args['kpt_thr'] and 
                        keypoint_scores[12] > call_args['kpt_thr'])
            
            if hip_valid:
                hip_mid_x = int((keypoints[11][0] + keypoints[12][0]) / 2)
                hip_mid_y = int((keypoints[11][1] + keypoints[12][1]) / 2)
                hip_mid_point = (hip_mid_x, hip_mid_y)
                
                # 绘制髂前上棘连线中点
                cv2.circle(display_frame, hip_mid_point, call_args['radius'], (165, 0, 255), -1)
            
            # 如果两个点都有效，绘制它们之间的连线
            if neck_valid and hip_valid:
                cv2.line(display_frame, neck_vertebra_point, hip_mid_point, (255, 255, 255), call_args['thickness'])
            
            # 只绘制需要的关键点
            for idx in kept_indices:
                if idx < len(keypoints) and keypoint_scores[idx] > call_args['kpt_thr']:
                    x, y = int(keypoints[idx][0]), int(keypoints[idx][1])
                    cv2.circle(display_frame, (x, y), call_args['radius'], (0, 255, 0), -1)
            
            # 绘制骨架连线
            for sk_idx, (start_idx, end_idx) in enumerate(skeleton):
                if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                    keypoint_scores[start_idx] > call_args['kpt_thr'] and 
                    keypoint_scores[end_idx] > call_args['kpt_thr']):
                    
                    start_pt = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                    end_pt = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                    
                    color = link_colors[sk_idx] if sk_idx < len(link_colors) else (0, 255, 0)
                    thickness = call_args['thickness']
                    cv2.line(display_frame, start_pt, end_pt, color, thickness)
            
            # 在头部上方绘制ID标签
            if keypoint_scores[0] > call_args['kpt_thr']:  # 如果鼻子关键点可见
                id_x = int(keypoints[0][0])
                id_y = int(keypoints[0][1] - 30)  # 在头部上方放置ID标签
                # 绘制黑色背景框增强可读性
                cv2.rectangle(display_frame, (id_x-20, id_y-15), (id_x+20, id_y+5), (0, 0, 0), -1)
                # 绘制ID文本
                cv2.putText(display_frame, f"ID:{track_id}", (id_x-18, id_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # 如果点数不够，只绘制可用关键点
            for kpt_idx, (kpt, score) in enumerate(zip(keypoints, keypoint_scores)):
                if kpt_idx in kept_indices and score > call_args['kpt_thr']:
                    x, y = int(kpt[0]), int(kpt[1])
                    cv2.circle(display_frame, (x, y), call_args['radius'], (0, 255, 0), -1)
    
    # 检索边界框（如果存在）
    bbox = instance.get('bbox', None)
    if bbox is not None and call_args['draw_bbox']:
        try:
            # 处理不同格式的边界框
            if isinstance(bbox, list) or isinstance(bbox, np.ndarray):
                if len(bbox) == 4:
                    # 标准格式 [x1, y1, x2, y2]
                    x1, y1, x2, y2 = [int(float(coord)) if isinstance(coord, (int, float, str)) else int(coord[0]) for coord in bbox]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # 在边界框左上角绘制ID标签
                    cv2.putText(display_frame, f"ID:{track_id}", (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                elif len(bbox) == 5:
                    # 带置信度的格式 [x1, y1, x2, y2, score]
                    x1, y1, x2, y2, _ = [int(float(coord)) if isinstance(coord, (int, float, str)) else int(coord[0]) for coord in bbox[:5]]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # 在边界框左上角绘制ID标签
                    cv2.putText(display_frame, f"ID:{track_id}", (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        except Exception as e:
            print(f"处理边界框时出错: {str(e)}, 边界框数据: {bbox}")

# 新增：进程间共享数据的结构
class SharedData:
    def __init__(self):
        # 创建共享变量用于通信
        self.running = mp.Value('b', True)

# 摄像头处理进程函数
def camera_process(camera_id, return_dict, shared_data, model_name='rtmpose-l_8xb32-270e_coco-wholebody-384x288', device='cuda:0'):
    """每个摄像头的独立处理进程
    
    Args:
        camera_id: 摄像头ID
        return_dict: 用于返回处理结果的字典
        shared_data: 进程间共享数据
        model_name: 模型名称
        device: 设备名称
    """
    try:
        # 设置CUDNN加速
        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
        
        # 初始化摄像头
        cap = cv2.VideoCapture(camera_id)
        
        # 检查摄像头是否成功打开
        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            return_dict[f'error_{camera_id}'] = f"无法打开摄像头 {camera_id}"
            return
            
        # 尝试设置更高的捕获分辨率和其他优化
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        # 设置缓冲区大小为1，减少延迟
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 在独立进程中初始化模型
        try:
            inferencer = MMPoseInferencer(
                pose2d=model_name,
                device=device,
                scope='mmpose',
                show_progress=False
            )
            print(f"进程 {camera_id} 成功加载模型到 {device}")
            
            # 预热模型以初始化CUDA核心和缓存
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            with torch.inference_mode(), torch.amp.autocast(device_type='cuda', enabled=True):
                for _ in range(10):  # 预热10次
                    _ = list(inferencer(dummy_frame))
                    
            # 强制同步GPU，确保预热完成    
            if 'cuda' in device:
                torch.cuda.synchronize()
                
        except Exception as e:
            print(f"进程 {camera_id} 模型加载失败: {str(e)}")
            return_dict[f'error_{camera_id}'] = f"模型加载失败: {str(e)}"
            return
            
        # 推理配置
        call_args = {
            'show': False,
            'draw_bbox': True,  # 设为True以显示边界框和ID
            'radius': 4,
            'thickness': 2,
            'kpt_thr': 0.4,     # 降低关键点阈值，增加检测灵敏度
            'bbox_thr': 0.3,    # 保持边界框阈值
            'nms_thr': 0.5,     # 降低NMS阈值，更容易检测到多人
            'pose_based_nms': True,
            'max_num_bboxes': 15  # 增加检测的最大人数上限
        }
        
        # 性能统计
        inference_times = []
        frame_count = 0
        start_time = time.time()
        
        # 创建图像转换缓存
        img_cache = None
        
        # 跟踪状态变量
        results_last = []  # 上一帧的结果
        next_id = 0        # 下一个可用的ID
        tracking_thr = 0.3  # IOU阈值
        
        # 自定义跟踪函数
        def track_by_iou(bbox, results_last, thr):
            """使用IOU跟踪对象"""
            max_iou_score = -1
            max_index = -1
            track_id = -1
            
            # 确保bbox格式正确
            if not isinstance(bbox, (list, np.ndarray)) or len(bbox) < 4:
                return -1, results_last
            
            for index, res_last in enumerate(results_last):
                last_bbox = res_last.get('bbox', None)
                if last_bbox is None or not isinstance(last_bbox, (list, np.ndarray)) or len(last_bbox) < 4:
                    continue
                
                # 计算IOU，使用MMPose提供的函数
                try:
                    # 使用MMPose的IOU计算函数
                    iou_score = _compute_iou(bbox, last_bbox)
                    if iou_score > max_iou_score:
                        max_iou_score = iou_score
                        max_index = index
                except Exception as e:
                    print(f"计算IOU时出错: {str(e)}, bbox1: {bbox}, bbox2: {last_bbox}")
                    continue
            
            # 如果IOU得分大于阈值，使用匹配的上一帧对象的ID
            if max_iou_score > thr and max_index != -1:
                track_id = results_last[max_index].get('track_id', -1)
                # 从结果列表中移除已匹配的项
                results_last.pop(max_index)
            
            return track_id, results_last
        
        # 主处理循环
        while shared_data.running.value:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 处理帧
            start_inference = time.time()
            
            # 优化：使用缓存分配的内存进行颜色转换，避免重复内存分配
            if img_cache is None or img_cache.shape != frame.shape:
                img_cache = np.empty(frame.shape, dtype=np.uint8)
            
            # 优化：直接使用cv2的转换，避免通过RGB中间格式
            # 注意：如果MMPose必须使用RGB格式，则保留原转换
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, img_cache)
            
            # 推理
            with torch.inference_mode(), torch.amp.autocast(device_type='cuda', enabled=True):
                raw_results = list(inferencer(img_cache, **call_args))
                # 确保GPU操作完成，减少延迟抖动
                if 'cuda' in device:
                    torch.cuda.synchronize()
                
            inference_time = time.time() - start_inference
            inference_times.append(inference_time)
            if len(inference_times) > 30:
                inference_times.pop(0)
            
            # 添加跟踪ID处理
            current_results = []
            for result in raw_results:
                pred_instances = result.get('predictions', [])
                
                # 遍历pred_instances中的所有实例
                if pred_instances and len(pred_instances) > 0:
                    # MMPose 2.x返回的结果结构可能是一个列表，包含多个模型的预测结果
                    # 我们需要确保正确处理这种结构
                    for instance_list in pred_instances:
                        # 检查instance_list是否为列表类型
                        if isinstance(instance_list, list):
                            # 遍历列表中的每个实例（每个人）
                            for instance in instance_list:
                                # 获取边界框
                                bbox = instance.get('bbox', None)
                                keypoints = instance.get('keypoints', None)
                                
                                # 如果没有边界框但有关键点，则使用关键点创建一个边界框
                                if bbox is None and keypoints is not None and len(keypoints) > 0:
                                    try:
                                        keypoints = np.array(keypoints)
                                        if keypoints.size > 0:
                                            # 计算包含所有关键点的边界框
                                            x_min = np.min(keypoints[:, 0])
                                            y_min = np.min(keypoints[:, 1])
                                            x_max = np.max(keypoints[:, 0])
                                            y_max = np.max(keypoints[:, 1])
                                            bbox = [x_min, y_min, x_max, y_max]
                                            instance['bbox'] = bbox
                                    except Exception as e:
                                        print(f"从关键点创建边界框时出错: {str(e)}")
                                
                                if bbox is not None:
                                    try:
                                        # 确保边界框格式正确
                                        if isinstance(bbox, (list, np.ndarray)) and len(bbox) >= 4:
                                            # 尝试跟踪
                                            track_id, results_last = track_by_iou(bbox, results_last, tracking_thr)
                                            if track_id == -1:
                                                # 如果没有匹配，分配新ID
                                                track_id = next_id
                                                next_id += 1
                                            
                                            # 设置跟踪ID
                                            instance['track_id'] = track_id
                                            
                                            # 保存当前实例以供下一帧使用
                                            current_results.append(instance)
                                    except Exception as e:
                                        print(f"处理跟踪ID时出错: {str(e)}, bbox: {bbox}")
                        elif isinstance(instance_list, dict):
                            # 如果直接是一个字典(单个人的情况)，直接处理
                            instance = instance_list
                            bbox = instance.get('bbox', None)
                            keypoints = instance.get('keypoints', None)
                            
                            # 如果没有边界框但有关键点，则使用关键点创建一个边界框
                            if bbox is None and keypoints is not None and len(keypoints) > 0:
                                try:
                                    keypoints = np.array(keypoints)
                                    if keypoints.size > 0:
                                        # 计算包含所有关键点的边界框
                                        x_min = np.min(keypoints[:, 0])
                                        y_min = np.min(keypoints[:, 1])
                                        x_max = np.max(keypoints[:, 0])
                                        y_max = np.max(keypoints[:, 1])
                                        bbox = [x_min, y_min, x_max, y_max]
                                        instance['bbox'] = bbox
                                except Exception as e:
                                    print(f"从关键点创建边界框时出错: {str(e)}")
                            
                            if bbox is not None:
                                try:
                                    # 确保边界框格式正确
                                    if isinstance(bbox, (list, np.ndarray)) and len(bbox) >= 4:
                                        # 尝试跟踪
                                        track_id, results_last = track_by_iou(bbox, results_last, tracking_thr)
                                        if track_id == -1:
                                            # 如果没有匹配，分配新ID
                                            track_id = next_id
                                            next_id += 1
                                        
                                        # 设置跟踪ID
                                        instance['track_id'] = track_id
                                        
                                        # 保存当前实例以供下一帧使用
                                        current_results.append(instance)
                                except Exception as e:
                                    print(f"处理跟踪ID时出错: {str(e)}, bbox: {bbox}")
            
            # 更新跟踪状态
            results_last = current_results
                
            # 处理结果并渲染到帧上
            display_frame = process_pose_results(frame, raw_results, call_args)
            
            # 计算FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # 每100帧重置计数器，防止数值过大
            if frame_count >= 100:
                start_time = time.time()
                frame_count = 0
                
            # 获取平均推理时间
            avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
            
            # 添加FPS和性能信息
            avg_inference_time_ms = avg_inference_time * 1000  # 转换为毫秒
            info_text = f"CAM {camera_id} | FPS: {fps:.1f} | Inference: {avg_inference_time_ms:.0f}ms"
            cv2.putText(display_frame, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 将结果存储在共享字典中
            # 由于多进程不能直接共享图像数据，需要编码为字节
            # 优化：降低JPEG质量以加快编码速度，同时保持视觉质量
            _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return_dict[f'frame_{camera_id}'] = buffer.tobytes()
            
            # 提前释放不需要的大型对象
            del raw_results
            
            # 为了调试输出检测到的人数
            if len(current_results) > 0:
                print(f"摄像头 {camera_id} 检测到 {len(current_results)} 人")
            
    except Exception as e:
        print(f"进程 {camera_id} 出错: {str(e)}")
        return_dict[f'error_{camera_id}'] = str(e)
        
    finally:
        # 释放资源
        if 'cap' in locals():
            cap.release()
        print(f"进程 {camera_id} 已退出")

def main():
    """主函数"""
    try:
        # 检测可用设备
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")
        
        # 创建多进程管理器
        manager = mp.Manager()
        return_dict = manager.dict()
        
        # 创建共享数据结构
        shared_data = SharedData()
        shared_data.running.value = True
        
        # 创建两个摄像头处理进程
        process1 = mp.Process(target=camera_process, args=(0, return_dict, shared_data, 'rtmpose-l_8xb32-270e_coco-wholebody-384x288', device))
        process2 = mp.Process(target=camera_process, args=(1, return_dict, shared_data, 'rtmpose-l_8xb32-270e_coco-wholebody-384x288', device))

        # process1 = mp.Process(target=camera_process, args=(0, return_dict, shared_data, 'rtmw-x_8xb320-270e_cocktail14-384x288', device))
        # process2 = mp.Process(target=camera_process, args=(1, return_dict, shared_data, 'rtmw-x_8xb320-270e_cocktail14-384x288', device))
        
        # 启动进程
        process1.start()
        process2.start()
        
        print("按'q'键退出")
        
        # 主循环 - 显示结果
        while True:
            # 检查是否有帧需要显示
            if 'frame_0' in return_dict:
                # 解码图像数据
                buffer = np.frombuffer(return_dict['frame_0'], dtype=np.uint8)
                display_frame1 = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                cv2.imshow('CAM0 - RTMPose', display_frame1)
                
            if 'frame_1' in return_dict:
                # 解码图像数据
                buffer = np.frombuffer(return_dict['frame_1'], dtype=np.uint8)
                display_frame2 = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                cv2.imshow('CAM1 - RTMPose', display_frame2)
                
            # 检查错误
            for cam_id in [0, 1]:
                if f'error_{cam_id}' in return_dict:
                    print(f"CAM {cam_id} Error: {return_dict[f'error_{cam_id}']}")
                    return_dict.pop(f'error_{cam_id}', None)
            
            # 检查退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            
    except Exception as e:
        print(f"主进程错误: {str(e)}")
        
    finally:
        # 设置退出标志
        shared_data.running.value = False
        
        # 等待进程结束
        if 'process1' in locals() and process1.is_alive():
            process1.join(timeout=1.0)
            if process1.is_alive():
                process1.terminate()
                
        if 'process2' in locals() and process2.is_alive():
            process2.join(timeout=1.0)
            if process2.is_alive():
                process2.terminate()
        
        # 关闭窗口
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == '__main__':
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main() 