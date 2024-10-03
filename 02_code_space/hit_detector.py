import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.spatial import distance
import matplotlib.pyplot as plt


class HitDetector:
    def __init__(self):
        pass

    # def predict(self, y_ball, smooth=True):
    #     """
    #     预测出所有击球帧，返回它们在视频帧中的序号

    #     :param y_ball: list of float or None
    #         视频中每一帧的网球 y 坐标列表。
    #     :param smooth: bool, optional
    #         是否对 y 坐标进行平滑处理。默认为 `True`。
    #     :return: set of int
    #         检测到的击球点对应的帧编号集合。
    #     """
    #     if smooth:
    #         y_ball = self.smooth_predictions(y_ball)

    #         # 对y_ball进行求导
    #         diff_y = np.gradient(y_ball)

    #         # 提取出所有导数为0的帧（y轴方向导数为0时是击球点）
    #         # 由于数值计算中导数可能不会精确为0，设定一个接近0的阈值
    #         threshold_derivative = 1e-3
    #         ind_hit = np.where(np.ab(diff_y) < threshold_derivative)[0]
            
    #         # 移除因平滑和插值引入的边缘点
    #         ind_hit = ind_hit[(ind_hit > 0) & (ind_hit < len(y_ball) - 1)]

    #         # 后处理, 过滤连续的击球点，保留局部最小值或最大值
    #         if len(ind_hit) > 0:
    #             ind_hit = self.postprocess(ind_hit, diff_y)

    #         num_frames = list(range(len(y_ball)))
    #         frames_hit = [num_frames[x] for x in ind_hit]

    #         return set(frames_hit)

    def predict(self, y_ball, smooth = True):
        if smooth:
            y_ball = self.smooth_predictions(y_ball)
            print(f"y_ball_after_smooth:{y_ball}")

        # 创建DataFrame
        df = pd.DataFrame({'mid_y':y_ball})

        # 应用滚动平均
        window_size = 5
        df['mid_y_rolling_mean'] = df['mid_y'].rolling(window=window_size, min_periods=1).mean()
        print(f"mid_y_rooling_mean{df['mid_y_rolling_mean']}")

        # 计算delta_y
        df['delta_y'] = df['mid_y_rolling_mean'].diff()

        # 初始化ball_hit列
        df['ball_hit'] = 0

        # 设置最小变化帧数
        minimum_change_frames_for_hit = 25

        # 遍历DataFrame以检测击球点
        for i in range(1, len(df) - int(minimum_change_frames_for_hit * 1.2)):
            # 检测delta_y的符号变化
            negative_change = df['delta_y'].iloc[i] > 0 and df['delta_y'].iloc[i+1] < 0
            positive_change = df['delta_y'].iloc[i] < 0 and df['delta_y'].iloc[i+1] > 0

            if negative_change or positive_change:
                change_count = 0
                for j in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    if j >= len(df):
                        break
                    if negative_change:
                        # 检测后续帧是否继续为负变化
                        if df['delta_y'].iloc[j] < 0 < df['delta_y'].iloc[i]:
                            change_count += 1
                    elif positive_change:
                        # 检测后续帧是否继续为正变化
                        if df['delta_y'].iloc[j] > 0 > df['delta_y'].iloc[i]:
                            change_count += 1

                if change_count > (minimum_change_frames_for_hit - 1):
                    df.at[i, 'ball_hit'] = 1
        
        # 提取击球帧编号
        frame_nums_with_ball_hits = df[df['ball_hit'] == 1].index.tolist()



        return set(frame_nums_with_ball_hits)

    def smooth_predictions(self, y_ball):
        """
        平滑 y 坐标序列，填补缺失值以便求导

        :param y_ball: list of float or None
            视频中每一帧的网球 y 坐标列表。
        :return: list of float
            平滑处理后的 y 坐标列表。
        """
        # 标记缺失值
        is_none = [int(y is None) for y in y_ball]
        # 填补缺失值
        interp = 5  # 插值窗口大小
        counter = 0
    
        for num in range(interp, len(y_ball) - 1):
            if y_ball[num] is None and sum(is_none[num- interp:num]) == 0 and counter < 3: # 如果y_ball无坐标且在窗口上下文中不存在为空的值
                # 使用插值方法预测缺失值
                y_ext = self.extraploate(y_ball[num - interp:num])
                y_ball[num] = y_ext
                is_none[num] = 0
                if y_ball[num+1]:
                    dist = abs(y_ext - y_ball[num + 1])
                    if dist > 56.5:
                        y_ball[num + 1], is_none[num + 1] = None, 1
                counter += 1
            else:
                counter = 0
        
        
        return y_ball


    def extraploate(self, y_coords):
        """
        使用三次样条插值方法预测缺失帧的 y 坐标

        :param y_coords: list of float
            前后帧的 y 坐标列表，用于插值。
        :return: float
            预测的 y 坐标。
        """
        xs = list(range(len(y_coords)))
        func_y = CubicSpline(xs, y_coords, bc_type='natural')
        y_ext = func_y(len(y_coords))
        return float(y_ext)
