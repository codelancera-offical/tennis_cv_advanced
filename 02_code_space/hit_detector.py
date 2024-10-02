import numpy as np
import pandas as pd

class HitDetector:
    def __init__(self):
        pass

    def predict(self, y_ball, smooth=True):
        """
        预测出所有击球帧，返回它们在视频帧中的序号
        param:
        y_ball: 视频中每一帧的网球y坐标
        """
        if smooth:
            y_ball = self.smooth_predictions(y_ball)
            # 对y_ball进行求导
            # diff_y_ball = 
            num_frames = len(y_ball)


    def smooth_predictions(self, y_ball):
        """
        平滑y坐标序列, 保证连续性以便求导
        """
        pass

    def extraploate(self, y_coords):
        """
        smooth_predictions 工具函数, 三次样条插值实现平滑
        """
        pass

