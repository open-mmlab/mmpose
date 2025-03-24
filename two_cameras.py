import cv2
import threading
import queue
import time
import numpy as np

# 全局变量控制程序退出
running = True

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
            print(f"无法打开摄像头 ID: {self.camera_id}")
            global running
            running = False
            return
        
        print(f"成功打开摄像头 ID: {self.camera_id}")
            
        # 主捕获循环
        while running:
            ret, frame = self.cap.read()
            if not ret:
                print(f"无法获取摄像头 {self.camera_id} 的视频帧")
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
            
    def get_frame(self, timeout: float = 0.1):
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

def display_camera_feed(cameras):
    """
    显示多个摄像头的画面
    
    Args:
        cameras: 摄像头对象列表
    """
    fps_time = time.time()
    frames_processed = 0
    
    while running:
        display_frames = []
        
        # 从每个摄像头获取帧
        for i, camera in enumerate(cameras):
            result = camera.get_frame()
            if result is not None:
                frame_id, frame = result
                # 在帧上添加相机ID和帧号
                cv2.putText(frame, f"Camera {i} - Frame {frame_id}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                display_frames.append(frame)
        
        # 只有当所有摄像头都有帧时才进行显示
        if len(display_frames) == len(cameras):
            frames_processed += 1
            
            # 计算并显示FPS
            current_time = time.time()
            if current_time - fps_time >= 1.0:  # 每秒更新一次FPS
                fps = frames_processed / (current_time - fps_time)
                fps_time = current_time
                frames_processed = 0
                print(f"FPS: {fps:.2f}")
            
            # 显示每个摄像头的画面
            for i, frame in enumerate(display_frames):
                cv2.imshow(f"Camera {i}", frame)
        
        # 检查用户是否按下ESC键退出
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键
            break

def main():
    """主函数"""
    global running
    
    try:
        # 创建两个摄像头捕获对象(使用摄像头ID 0和1)
        camera0 = CameraCapture(camera_id=0)
        camera1 = CameraCapture(camera_id=1)
        
        cameras = [camera0, camera1]
        
        # 显示摄像头画面
        display_camera_feed(cameras)
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        # 程序退出时清理资源
        running = False
        cv2.destroyAllWindows()
        
        # 等待线程结束
        time.sleep(1)
        
        print("程序已退出")

if __name__ == "__main__":
    main() 