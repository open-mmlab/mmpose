# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import time
import torch
import threading
import queue
from mmpose.apis.inferencers import MMPoseInferencer
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional

# 全局变量控制程序退出
running = True

# 使用上下文管理器进行CUDA异步处理
@contextmanager
def torch_inference_mode():
    """使用torch推理模式的上下文管理器，优化推理性能"""
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
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
            'radius': 4,
            'thickness': 2,
            'kpt_thr': 0.5,
            'bbox_thr': 0.3,
            'nms_thr': 0.65,  # 对于rtmpose，更高的NMS阈值通常更好
            'pose_based_nms': True  # 启用基于姿态的NMS
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

    for result in results:
        pred_instances = result.get('predictions', [])
        
        # 如果有预测结果，在原始帧上绘制
        if pred_instances and len(pred_instances) > 0:
            for instance in pred_instances:
                # 只处理第一个模型的结果（通常是人体姿态）
                if isinstance(instance, list) and len(instance) > 0:
                    instance = instance[0]
                    
                    # 获取关键点和得分
                    keypoints = instance.get('keypoints', None)
                    keypoint_scores = instance.get('keypoint_scores', None)
                    
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
                                elif len(bbox) == 5:
                                    # 带置信度的格式 [x1, y1, x2, y2, score]
                                    x1, y1, x2, y2, _ = [int(float(coord)) if isinstance(coord, (int, float, str)) else int(coord[0]) for coord in bbox[:5]]
                                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        except Exception as e:
                            print(f"处理边界框时出错: {str(e)}, 边界框数据: {bbox}")
    
    return display_frame

def main():
    """主函数"""
    try:
        global running
        running = True
        
        # 检测可用设备
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")
        
        # 创建帧捕获器
        camera = CameraCapture(camera_id=0, queue_size=4)
        
        # 创建帧处理器
        processor = FrameProcessor(
            model_name='rtmpose-l_8xb32-270e_coco-wholebody-384x288',
            device=device,
            input_queue_size=3,
            output_queue_size=3
        )
        
        print("按'q'键退出")
        
        # FPS计算相关变量
        frame_count = 0
        start_time = time.time()
        fps = 0
        display_fps = 0
        
        # 最后显示的帧，用于在没有新结果时保持显示
        last_display_frame = None
        
        while running:
            # 获取最新帧用于处理
            frame_data = camera.get_frame(timeout=0.01)
            if frame_data:
                frame_id, frame = frame_data
                # 添加到处理队列
                processor.add_frame(frame_id, frame)
            
            # 获取处理结果
            result_data = processor.get_result(timeout=0.01)
            
            # 显示处理后的帧
            if result_data:
                _, frame, results, inference_time = result_data
                
                # 处理结果并显示
                display_frame = process_pose_results(frame, results, processor.call_args)
                last_display_frame = display_frame
                
                # 更新FPS计算
                frame_count += 1
                if frame_count >= 10:  # 每10帧更新一次FPS
                    end_time = time.time()
                    display_fps = frame_count / (end_time - start_time)
                    frame_count = 0
                    start_time = time.time()
                
                # 添加FPS和性能信息
                avg_inference_time = processor.get_avg_inference_time() * 1000  # 转换为毫秒
                info_text = f"FPS: {display_fps:.1f} | Inference: {avg_inference_time:.0f}ms"
                cv2.putText(display_frame, info_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 显示结果
                cv2.imshow('RTMPose Real-time Pose Estimation', display_frame)
            elif last_display_frame is not None:
                # 如果没有新结果但有之前的帧，则继续显示上一帧
                cv2.imshow('RTMPose Real-time Pose Estimation', last_display_frame)
            
            # 检查是否按下'q'键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break
    
    except Exception as e:
        print(f"程序运行错误: {str(e)}")
    
    finally:
        # 设置退出标志
        running = False
        
        # 等待线程自然退出
        time.sleep(0.5)
        
        # 清理资源
        if 'camera' in locals():
            camera.release()
            
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == '__main__':
    main() 