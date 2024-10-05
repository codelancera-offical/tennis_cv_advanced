from tracknet import BallTrackerNet
import torch
import cv2
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm

class BallDetector:
    def __init__(self, path_model=None, device='cuda'):
        self.model = BallTrackerNet(input_channels=9, out_channels=256)
        self.device = device
        if path_model:
            self.model.load_state_dict(torch.load(path_model, map_location=device))
            self.model = self.model.to(device)
            self.model.eval()
        self.width = 640
        self.height = 360

    def infer_model(self, frames):
        """ Run pretrained model on a consecutive list of frames
            在连续的视频帧中运行预训练模型, 检测并跟踪网球的位置
        :params
            frames: list of consecutive video frames
        :return
            ball_track: list of detected ball points
            列表，包含每一帧中检测到的网球坐标 (x, y) 如果未检测到，则为 (None, None)
        """
        ball_track = [(None, None)]*2
        prev_pred = [None, None]
        for num in tqdm(range(2, len(frames))):

            # 图像预处理：将当前帧及其前两帧调整为统一的尺寸(640 x 360),
            img = cv2.resize(frames[num], (self.width, self.height))
            img_prev = cv2.resize(frames[num-1], (self.width, self.height))
            img_preprev = cv2.resize(frames[num-2], (self.width, self.height))

            # 将当前帧与前两帧在通道维度上拼接，形成具有9个通道的输入
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32)/255.0    # 归一化处理
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            # 模型推理
            out = self.model(torch.from_numpy(inp).float().to(self.device))
            output = out.argmax(dim=1).detach().cpu().numpy()   # 输出特征图

            # 从特征图中提取网球的坐标
            x_pred, y_pred = self.postprocess(output, prev_pred)
            prev_pred = [x_pred, y_pred]
            ball_track.append((x_pred, y_pred))
        return ball_track

    def postprocess(self, feature_map, prev_pred, scale=2, max_dist=80):
        """
        对模型的特征图进行后处理，提取网球的精确坐标
        :params
            feature_map: feature map with shape (1,360,640)
                模型输出的特征图, 形状为(1, 360, 640)
            prev_pred: [x,y] coordinates of ball prediction from previous frame
                上一帧的网球距离坐标[x, y], 用于距离过滤
            scale: scale for conversion to original shape (720,1280)
                将检测到的坐标缩放到原始图像尺寸的比例因子, 默认为2(扩大到720p,  3扩大到1080p)
            max_dist: maximum distance from previous ball detection to remove outliers
                上一帧坐标允许的最大距离, 用于过滤异常值检测, 默认值为80 
        :return
            (x,y) ball coordinates in tuple form, 未检测到则为None
        """

        # 将特征图的像素值放大到[0, 255]范围，并且条状形状为(360, 640)，并转换为uint8类型以供后续图像处理
        feature_map *= 255
        feature_map = feature_map.reshape((self.height, self.width))
        feature_map = feature_map.astype(np.uint8)

        # 应用二值化阈值，将特征图转换为2值热图，突出网球的位置
        ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        
        # 霍夫圆变换检测， 使用霍夫圆变换检测热图中的圆形区域，假设网球呈现为小圆形
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                                   maxRadius=7)
       
       
        x, y = None, None
        if circles is not None: # 如果检测到圆形区域
            if prev_pred[0]:    # 且存在上一帧的预测坐标prev_pred
                for i in range(len(circles[0])):    # 计算当前检测到的圆与上一帧的预测坐标之间的欧几里得距离
                    x_temp = circles[0][i][0]*scale
                    y_temp = circles[0][i][1]*scale
                    dist = distance.euclidean((x_temp, y_temp), prev_pred)
                    if dist < max_dist: # 如果距离小于max_dist， 确定为当前帧的网球位置
                        x, y = x_temp, y_temp
                        break                
            else:   # 如果不存在上一帧，直接选择第一个检测到的圆作为网球位置
                x = circles[0][0][0]*scale
                y = circles[0][0][1]*scale
        return x, y
