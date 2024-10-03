# 第三方库
import cv2
import numpy as np
import argparse
import torch
import json


from court_detection_net import CourtDetectorNet
from court_reference import CourtReference
from bounce_detector import BounceDetector
from person_detector import PersonDetector
from ball_detector import BallDetector
from hit_detector import HitDetector
from info_panel import HitInfoPanel
from utils import scene_detect

# main-utils
def read_video(path_video) -> tuple[list, int]:
    """
    读取指定路径的视频文件，提取所有帧并获取视频帧率
    """
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break    
    cap.release()
    return frames, fps

def write(imgs_res, fps, path_output_video) -> None:
    """将处理后的图像帧列表写入视频文件"""
    height, width = imgs_res[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for num in range(len(imgs_res)):
        frame = imgs_res[num]
        out.write(frame)
    out.release()    

def get_court_img() -> np.ndarray:
    """生成一个用于表示网球场区域的RGB参照图像(mini-court), 该图像在后续处理中将检测到的物体（如球和球员）映射到小地图中，从而直观地展示物体相对于球场的位置分布。"""
    court_reference = CourtReference()
    court = court_reference.build_court_reference()
    court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court, court, court), axis=2)*255).astype(np.uint8)
    return court_img


def main(frames, scenes, bounces, ball_track, homography_matrices, kps_court, persons_top, persons_bottom,
         draw_trace=False, trace=10, hits=None, fps=None, ball_track_smooth=None):
    """功能: 整合所有信息, 处理视频帧，绘制网球轨迹, keypoints, 球员检测框以及弹跳点，并生成带有可视化结果的图像列表"""
    """
    :params
        // well-functioning
        frames: list of original images                                 原始视频帧列表
        scenes: list of beginning and ending of video fragment          视频片段的起始和结束帧编号列表
        bounces: list of image numbers where ball touches the ground    球触地的帧编号列表
        ball_track: list of (x,y) ball coordinates                      每帧中网球的(x, y)坐标列表
        homography_matrices: list of homography matrices                每帧的单应性矩阵列表，用于透视变换
        kps_court: list of 14 key points of tennis court                每一帧中网球场地的14个关键点列表
        persons_top: list of person bboxes located in the top of tennis court       每帧中位于网球场上半场的球员的检测框列表
        persons_bottom: list of person bboxes located in the bottom of tennis court 每帧中位于网球场下半场的球员的检测框列表
        draw_trace: whether to draw ball trace                          是否绘制网球轨迹, 默认值为False
        trace: the length of ball trace                                 网球轨迹的长度(帧数), 默认值为7

        // wait to add
        person_top_hits: list of image numbers where ball hitted by top_player.
        person_bottom_hits: list of image numbers where ball hitted by bottom_player.

    :return
        imgs_res: list of resulting images                              处理后带有可视化结果的图像帧列表
    """
    imgs_res = []
    width_minimap = 166
    height_minimap = 350
    is_track = [x is not None for x in homography_matrices]
    
    # added
    bounce_info = []    
    hit_info = []

    # 场景处理：对每个视频片段进行处理（一个网球视频中有不是拍球场的片段）
    for num_scene in range(len(scenes)):
        sum_track = sum(is_track[scenes[num_scene][0]:scenes[num_scene][1]])
        len_track = scenes[num_scene][1] - scenes[num_scene][0]
        eps = 1e-15
        scene_rate = sum_track/(len_track+eps)

        # 如果该片段场景中大部分帧存在单应性矩阵，处理该场景
        if (scene_rate > 0.5):
            court_img = get_court_img() # 生成网球场参考图像（mini court)
            hit_info_panel = HitInfoPanel()

            # 逐帧处理视频片段中的每一帧
            for i in range(scenes[num_scene][0], scenes[num_scene][1]):
                img_res = frames[i]
                inv_mat = homography_matrices[i]

                # 绘制网球轨迹 draw ball trajectory
                if ball_track[i][0]:
                    if draw_trace:  # 如果要绘制拖尾效果
                        for j in range(0, trace):
                            if i-j >= 0:
                                if ball_track[i-j][0]:
                                    draw_x = int(ball_track[i-j][0])
                                    draw_y = int(ball_track[i-j][1])
                                    img_res = cv2.circle(frames[i], (draw_x, draw_y),
                                    radius=3, color=(0, 255, 0), thickness=2)
                    
                    else:           # 如果不绘制拖尾效果
                        img_res = cv2.circle(img_res , (int(ball_track[i][0]), int(ball_track[i][1])), radius=5,
                                             color=(0, 255, 0), thickness=2)
                        img_res = cv2.putText(img_res, 'ball', 
                              org=(int(ball_track[i][0]) + 8, int(ball_track[i][1]) + 8),
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                              fontScale=0.8,
                              thickness=2,
                              color=(0, 255, 0))

                # 绘制球场关键点
                if kps_court[i] is not None:
                    for j in range(len(kps_court[i])):
                        img_res = cv2.circle(img_res, (int(kps_court[i][j][0, 0]), int(kps_court[i][j][0, 1])),
                                          radius=0, color=(0, 0, 255), thickness=10)

                height, width, _ = img_res.shape

                #----------弹跳-击球检测阶段----------#

                # 如果这一帧是弹跳（落地）帧，在小地图上绘制球的弹跳点
                if i in bounces and inv_mat is not None:
                    ball_point = ball_track[i]
                    ball_point_np = np.array(ball_point, dtype=np.float32).reshape(1, 1, 2)
                    ball_point_transformed = cv2.perspectiveTransform(ball_point_np, inv_mat)

                    # TODO: 保存弹跳点信息
                    bounce_info.append({
                        "frame": i,
                        "coordinates": (ball_track[i][0], ball_track[i][1]),
                        "coordinates_mini_court":(
                            int(ball_point_transformed[0, 0, 0]),
                            int(ball_point_transformed[0, 0, 1])
                        )
                    })

                    # 迷你球场绘制落点
                    court_img = cv2.circle(court_img, (int(ball_point_transformed[0, 0, 0]), int(ball_point_transformed[0, 0, 1])),
                                                       radius=0, color=(0, 255, 255), thickness=50)
                
                # 如果有击球信息
                if i in hits and inv_mat is not None:
                    # 获取这一击球帧和两帧之后的网球在视频中的位置
                    hit_point = ball_track_smooth[i]
                    print(f"hit_point:{hit_point}")

                    if i + 20 < len(ball_track):
                        after_20_frame_ball_point = ball_track[i + 20]
                        after_20_frame_ball_point_np = np.array(after_20_frame_ball_point, dtype=np.float32).reshape(1, 1, 2)


                    # 转换到mini_court坐标上
                    hit_point_mini = cv2.perspectiveTransform(np.array(hit_point, dtype=np.float32).reshape(1, 1, 2), inv_mat)
                    after_20_frame_ball_point_mini = cv2.perspectiveTransform(after_20_frame_ball_point_np, inv_mat)

                    # 计算球速
                    dx = after_20_frame_ball_point_mini[0, 0, 0] - hit_point_mini[0, 0, 0]
                    dy = after_20_frame_ball_point_mini[0, 0, 1] - hit_point_mini[0, 0, 1]
                    distance_pixels = np.sqrt(dx**2 + dy**2)

                    print(fps)

                    time_seconds = 20 / fps
                    speed_pixels_per_second = distance_pixels / time_seconds
                    
                    # 设置比例尺
                    pixels_per_meter = 152  # 152个像素素代表1米
                    speed_meters_per_second = speed_pixels_per_second / pixels_per_meter
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    # 确定击球者
                    hitter = "unknown"
                    min_distance = float('inf')
                    for label, persons in [("top", persons_top[i]), ("bottom", persons_bottom[i])]:
                        for person in persons:
                            person_bbox = person[0]
                            person_center = (
                                (person_bbox[0] + person_bbox[2]) / 2,
                                (person_bbox[1] + person_bbox[3]) / 2
                            )
                            print(f"person_center: {person_center}")
                            distance = np.sqrt((float(hit_point[0]) - float(person_center[0]))**2 + (float(hit_point[1]) - float(person_center[1]))**2)
                            if distance < min_distance:
                                min_distance = distance
                                hitter = label


                    # 确定击球姿势(TODO,先默认shoot)
                    pose = "shot"

                    # TODO:刷新击球看板上的数据，更新击球和姿态信息，姿态信息默认为shot
                    hit_info_panel.set_speed(speed_km_per_hour)
                    hit_info_panel.set_action(pose)
                    

                    # 保存击球信息
                    hit_info.append({
                        "frame": i,
                        "coordinates": hit_point,
                        "coordinates_mini_court":(
                            int(hit_point_mini[0, 0, 0]),
                            int(hit_point_mini[0, 0, 1])
                        ),
                        "speed_km_per_hour": speed_km_per_hour,
                        "speed_meters_per_second": speed_meters_per_second,
                        "hitter": hitter,
                        "pose": pose
                    })
                



                minimap = court_img.copy()

                # draw persons 绘制球员检测框
                persons = persons_top[i] + persons_bottom[i]                    
                for j, person in enumerate(persons):
                    if len(person[0]) > 0:
                        person_bbox = list(person[0])
                        img_res = cv2.rectangle(img_res, (int(person_bbox[0]), int(person_bbox[1])),
                                                (int(person_bbox[2]), int(person_bbox[3])), [255, 0, 0], 2)

                        # transmit person point to minimap 转换玩家坐标到mini court上
                        person_point = list(person[1])
                        person_point = np.array(person_point, dtype=np.float32).reshape(1, 1, 2)
                        person_point = cv2.perspectiveTransform(person_point, inv_mat)
                        minimap = cv2.circle(minimap, (int(person_point[0, 0, 0]), int(person_point[0, 0, 1])),
                                                           radius=0, color=(255, 0, 0), thickness=80)

                # 绘制迷你球场
                minimap = cv2.resize(minimap, (width_minimap, height_minimap))
                img_res[30:(30 + height_minimap), (width - 30 - width_minimap):(width - 30), :] = minimap   # 把图像的右上角区域绘制成minimap
                
                # 绘制看板
                panel_height, panel_width, _ = hit_info_panel.panel.shape
                # 计算新的位置
                panel_y_start = 30 + height_minimap + 20
                panel_y_end = panel_y_start + panel_height
                panel_x_start = width - 30 - panel_width
                panel_x_end = width - 30

                # 确保不会超出图像边界
                if panel_y_end + 20 > height:
                    panel_y_end = height - 20
                    panel_y_start = panel_y_end - panel_height

                img_res[panel_y_start:panel_y_end, panel_x_start:panel_x_end] = hit_info_panel.panel

                imgs_res.append(img_res)
                


        else:    
            imgs_res = imgs_res + frames[scenes[num_scene][0]:scenes[num_scene][1]]

    return imgs_res, bounce_info, hit_info
 


if __name__ == '__main__':

    """
    参数解析：通过 argparse 解析命令传入的函数
    使用例: python detect.py --path_ball_track_model "path/to/ball_track_model.pth" --path_court_model "path/to/court_model.pth" --path_bounce_model "path/to/bounce_model.pth" --path_input_video "path/to/input_video.mp4" --path_output_video "path/to/output_video.mp4"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_ball_track_model', type=str, help='path to pretrained model for ball detection')
    parser.add_argument('--path_court_model', type=str, help='path to pretrained model for court detection')
    parser.add_argument('--path_bounce_model', type=str, help='path to pretrained model for bounce detection')
    parser.add_argument('--path_input_video', type=str, help='path to input video')
    parser.add_argument('--path_output_video', type=str, help='path to output video')
    args = parser.parse_args()
    
    # 确定用于视觉检测的设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 视频读取 + 场景切割（视角大切换，内容大变换为一个scenes）
    print('reading video...\n')
    frames, fps = read_video(args.path_input_video) 
    print('spliting scenes...\n')
    scenes = scene_detect(args.path_input_video)    

    #----------检测阶段----------#

    # 球检测
    print('ball detection...\n')
    ball_detector = BallDetector(args.path_ball_track_model, device)
    ball_track = ball_detector.infer_model(frames)  # 获得每帧中的球坐标列表

    # 场地检测
    print('court detection...\n')
    court_detector = CourtDetectorNet(args.path_court_model, device)
    # 对视频每一帧生成单应性矩阵（如果有的话）以及关键点在视频中的坐标（如果有的话）
    homography_matrices, kps_court = court_detector.infer_model(frames)

    # 人物检测
    print('person detection...\n')
    person_detector = PersonDetector(device)
    # 检测出上半场球员和下半场球员在视频中的每一帧的坐标框坐标
    persons_top, persons_bottom = person_detector.track_players(frames, homography_matrices, filter_players=False)

    # bounce detection 弹跳（落地）检测
    print('bounce detection...\n')
    bounce_detector = BounceDetector(args.path_bounce_model)
    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces, ball_track_smooth = bounce_detector.predict(x_ball, y_ball)   # 球触地的帧编号列表
    print(f"ball_track_smooth:{ball_track_smooth}")

    # hit detection 击球检测
    print('hit detection...\n')
    hit_detector = HitDetector()
    hits = hit_detector.predict(y_ball)                 # 击球的帧编号列表


    # 绘制输出视频图像
    print('processing frames...\n')
    imgs_res, bounce_info, hit_info = main(frames, scenes, bounces, ball_track, homography_matrices, kps_court, persons_top, persons_bottom,
                    draw_trace=True, hits = hits, fps=fps, ball_track_smooth=ball_track_smooth)

    # 将bounce_info和hit_info写入文件
    print('writing info to files...\n')
    with open(f'./data/bounce_info.json', 'w') as f:
        json.dump(bounce_info, f, indent = 4)
    
    with open(f'./data/hit_info.json', 'w') as f:
        json.dump(hit_info, f, indent = 4)


    # 将视频写入文件路径
    print('writing into video...')
    write(imgs_res, fps, args.path_output_video)





