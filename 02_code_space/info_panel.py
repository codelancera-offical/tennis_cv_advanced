import cv2
import numpy as np

class HitInfoPanel:
    def __init__(self, width=300, height=80):  # 缩小宽度和高度
        self.panel_width = width
        self.panel_height = height
        self.panel = self.create_panel()
        self.pose = "-"
        self.speed = 0.0
    
    def create_panel(self) -> np.ndarray:
        """
        创建一个用于表示击球信息的看板, 包括击球姿态和球速信息
        """
        panel = np.zeros((self.panel_height, self.panel_width, 3), dtype=np.uint8)
        panel[:] = (50, 50, 50)  # 灰色背景
        cv2.line(panel, (self.panel_width//2, 0), (self.panel_width//2, self.panel_height), (255, 255, 255), 2)
        # 初始文本，调整字体大小为 0.5
        cv2.putText(panel, 'Pose: -', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, 'Speed: - km/h', (self.panel_width//2 + 10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return panel
    
    def update_panel(self):
        """
        根据当前的姿态和速度信息更新面板显示
        """
        # 重置面板
        self.panel[:] = (50, 50, 50)
        cv2.line(self.panel, (self.panel_width//2, 0), (self.panel_width//2, self.panel_height), (255, 255, 255), 2)
        # 绘制姿态，字体大小为 0.5
        cv2.putText(self.panel, f'{self.pose}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)
        # 绘制速度，字体大小为 0.5
        cv2.putText(self.panel, f'{self.speed} km/h', (self.panel_width//2 + 10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def set_speed(self, speed: float):
        self.speed = f"{speed:.2f}"
        self.update_panel()
    
    def set_action(self, action: str):
        self.pose = action
        self.update_panel()
