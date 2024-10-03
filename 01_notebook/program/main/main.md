# 网球视觉识别系统主程序说明文档

本文档详细介绍了网球视觉识别系统的主程序，包括各个函数的功能、输入输出参数以及整体工作流程。该系统旨在通过视频分析实现对网球比赛中球员、网球轨迹、弹跳点等的检测与跟踪，并生成带有可视化结果的输出视频。

## 目录

1. [概述](#1-概述)
2. [依赖库与模块](#2-依赖库与模块)
3. [函数说明](#3-函数说明)
   - [read_video](#read_video)
   - [get_court_img](#get_court_img)
   - [main](#main)
   - [write](#write)
4. [主程序流程](#4-主程序流程)
5. [命令行参数说明](#5-命令行参数说明)
6. [使用示例](#6-使用示例)
7. [注意事项](#7-注意事项)
8. [参考资料](#8-参考资料)

---

## 1. 概述

`main.py` 是网球视觉识别系统的主程序，负责整合各个检测与分析模块，处理输入的视频文件，进行目标检测、跟踪、动作识别等操作，最终生成带有可视化结果的输出视频。系统主要功能包括：

- **视频读取与预处理**：加载输入视频并提取帧信息。
- **球检测与跟踪**：检测网球的位置并追踪其运动轨迹。
- **场地检测与校正**：识别网球场地并进行透视变换以校正视角。
- **人物检测与跟踪**：检测并跟踪场上的球员。
- **弹跳点检测**：识别网球在地面的弹跳点。
- **结果可视化**：在视频帧上绘制检测结果和跟踪轨迹，并生成最终输出视频。

---

## 2. 依赖库与模块

### 2.1 编程语言

- **Python 3.6+**

### 2.2 第三方库

- **OpenCV (`cv2`)**：用于图像和视频处理。
- **NumPy (`numpy`)**：用于数值计算和数组操作。
- **Argparse (`argparse`)**：用于解析命令行参数。
- **PyTorch (`torch`)**：用于加载和运行深度学习模型。

### 2.3 自定义模块

- **court_detection_net**：网球场地检测模块。
- **court_reference**：网球场地参考图生成模块。
- **bounce_detector**：网球弹跳点检测模块。
- **person_detector**：球员检测与跟踪模块。
- **ball_detector**：网球检测与跟踪模块。
- **utils**：包含辅助函数，如场景检测 (`scene_detect`)。

---

## 3. 函数说明

### read_video

```python
def read_video(path_video):
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
```

#### 功能

读取指定路径的视频文件，提取所有帧并获取视频的帧率。

#### 输入参数

- `path_video` (str): 输入视频文件的路径。

#### 输出

- `frames` (list of np.ndarray): 包含视频所有帧的列表，每帧为一个 NumPy 数组。
- `fps` (int): 视频的帧率（每秒帧数）。

---

### get_court_img [[func-get_court_img]]

```python
def get_court_img():
    court_reference = CourtReference()
    court = court_reference.build_court_reference()
    court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court, court, court), axis=2)*255).astype(np.uint8)
    return court_img
```

#### 功能

生成网球场地的参考图像，用于后续的可视化展示。

#### 输入参数

无。

#### 输出

- `court_img` (np.ndarray): 生成的网球场地参考图像，尺寸为三通道彩色图。

---

### main

```python
def main(frames, scenes, bounces, ball_track, homography_matrices, kps_court, persons_top, persons_bottom,
         draw_trace=False, trace=7):
    """
    :params
        frames: list of original images
        scenes: list of beginning and ending of video fragment
        bounces: list of image numbers where ball touches the ground
        ball_track: list of (x,y) ball coordinates
        homography_matrices: list of homography matrices
        kps_court: list of 14 key points of tennis court
        persons_top: list of person bboxes located in the top of tennis court
        persons_bottom: list of person bboxes located in the bottom of tennis court
        draw_trace: whether to draw ball trace
        trace: the length of ball trace
    :return
        imgs_res: list of resulting images
    """
    imgs_res = []
    width_minimap = 166
    height_minimap = 350
    is_track = [x is not None for x in homography_matrices] 
    for num_scene in range(len(scenes)):
        sum_track = sum(is_track[scenes[num_scene][0]:scenes[num_scene][1]])
        len_track = scenes[num_scene][1] - scenes[num_scene][0]

        eps = 1e-15
        scene_rate = sum_track/(len_track+eps)
        if (scene_rate > 0.5):
            court_img = get_court_img()

            for i in range(scenes[num_scene][0], scenes[num_scene][1]):
                img_res = frames[i]
                inv_mat = homography_matrices[i]

                # draw ball trajectory
                if ball_track[i][0]:
                    if draw_trace:
                        for j in range(0, trace):
                            if i-j >= 0:
                                if ball_track[i-j][0]:
                                    draw_x = int(ball_track[i-j][0])
                                    draw_y = int(ball_track[i-j][1])
                                    img_res = cv2.circle(frames[i], (draw_x, draw_y),
                                    radius=3, color=(0, 255, 0), thickness=2)
                    else:    
                        img_res = cv2.circle(img_res , (int(ball_track[i][0]), int(ball_track[i][1])), radius=5,
                                             color=(0, 255, 0), thickness=2)
                        img_res = cv2.putText(img_res, 'ball', 
                              org=(int(ball_track[i][0]) + 8, int(ball_track[i][1]) + 8),
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                              fontScale=0.8,
                              thickness=2,
                              color=(0, 255, 0))

                # draw court keypoints
                if kps_court[i] is not None:
                    for j in range(len(kps_court[i])):
                        img_res = cv2.circle(img_res, (int(kps_court[i][j][0, 0]), int(kps_court[i][j][0, 1])),
                                          radius=0, color=(0, 0, 255), thickness=10)

                height, width, _ = img_res.shape

                # draw bounce in minimap
                if i in bounces and inv_mat is not None:
                    ball_point = ball_track[i]
                    ball_point = np.array(ball_point, dtype=np.float32).reshape(1, 1, 2)
                    ball_point = cv2.perspectiveTransform(ball_point, inv_mat)
                    court_img = cv2.circle(court_img, (int(ball_point[0, 0, 0]), int(ball_point[0, 0, 1])),
                                                       radius=0, color=(0, 255, 255), thickness=50)

                minimap = court_img.copy()

                # draw persons
                persons = persons_top[i] + persons_bottom[i]                    
                for j, person in enumerate(persons):
                    if len(person[0]) > 0:
                        person_bbox = list(person[0])
                        img_res = cv2.rectangle(img_res, (int(person_bbox[0]), int(person_bbox[1])),
                                                (int(person_bbox[2]), int(person_bbox[3])), [255, 0, 0], 2)

                        # transmit person point to minimap
                        person_point = list(person[1])
                        person_point = np.array(person_point, dtype=np.float32).reshape(1, 1, 2)
                        person_point = cv2.perspectiveTransform(person_point, inv_mat)
                        minimap = cv2.circle(minimap, (int(person_point[0, 0, 0]), int(person_point[0, 0, 1])),
                                                           radius=0, color=(255, 0, 0), thickness=80)

                minimap = cv2.resize(minimap, (width_minimap, height_minimap))
                img_res[30:(30 + height_minimap), (width - 30 - width_minimap):(width - 30), :] = minimap
                imgs_res.append(img_res)

        else:    
            imgs_res = imgs_res + frames[scenes[num_scene][0]:scenes[num_scene][1]] 
    return imgs_res        
```

#### 功能

处理视频帧，绘制网球轨迹、场地关键点、球员检测框和弹跳点，并生成带有可视化结果的图像列表。

#### 输入参数

- `frames` (list of np.ndarray): 原始视频帧列表。
- `scenes` (list of tuples): 视频片段的起始和结束帧编号列表。
- `bounces` (list of int): 球触地的帧编号列表。
- `ball_track` (list of tuples): 每帧中网球的 (x, y) 坐标列表。
- `homography_matrices` (list of np.ndarray or None): 每帧的单应性矩阵列表，用于透视变换。
- `kps_court` (list of list of np.ndarray or None): 每帧中网球场地的14个关键点列表。
- `persons_top` (list of list of tuples): 每帧中位于网球场上方球员的检测框列表。
- `persons_bottom` (list of list of tuples): 每帧中位于网球场下方球员的检测框列表。
- `draw_trace` (bool, optional): 是否绘制网球轨迹。默认值为 `False`。
- `trace` (int, optional): 网球轨迹的长度（帧数）。默认值为 `7`。

#### 输出

- `imgs_res` (list of np.ndarray): 处理后带有可视化结果的图像帧列表。

---

### write

```python
def write(imgs_res, fps, path_output_video):
    height, width = imgs_res[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for num in range(len(imgs_res)):
        frame = imgs_res[num]
        out.write(frame)
    out.release()
```

#### 功能

将处理后的图像帧列表写入输出视频文件。

#### 输入参数

- `imgs_res` (list of np.ndarray): 处理后的图像帧列表。
- `fps` (int): 输出视频的帧率。
- `path_output_video` (str): 输出视频文件的保存路径。

#### 输出

无（直接生成视频文件）。

---

## 4. 主程序流程

主程序通过命令行参数接收输入和输出路径，并依次执行以下步骤：

1. **参数解析**：通过 `argparse` 解析命令行传入的参数。
2. **环境设置**：检查是否有可用的 GPU，并设置设备（`cuda` 或 `cpu`）。
3. **视频读取**：调用 `read_video` 函数读取输入视频的所有帧及其帧率。
4. **场景检测**：使用 `scene_detect` 函数分析视频，分割为不同的场景片段。
5. **球检测**：
   - 初始化 `BallDetector` 模型并加载预训练权重。
   - 对所有帧进行球检测，生成网球轨迹 `ball_track`。
6. **场地检测**：
   - 初始化 `CourtDetectorNet` 模型并加载预训练权重。
   - 对所有帧进行场地检测，生成单应性矩阵 `homography_matrices` 和场地关键点 `kps_court`。
7. **人物检测**：
   - 初始化 `PersonDetector` 模型。
   - 对所有帧进行球员检测与跟踪，生成 `persons_top` 和 `persons_bottom`。
8. **弹跳检测**：
   - 初始化 `BounceDetector` 模型。
   - 根据球的轨迹 `ball_track` 预测弹跳点 `bounces`。
9. **结果处理**：
   - 调用 `main` 函数，传入所有检测结果，生成带有可视化结果的图像帧列表 `imgs_res`。
10. **视频写入**：
    - 调用 `write` 函数，将 `imgs_res` 写入输出视频文件。

---

## 5. 命令行参数说明

主程序通过命令行接收以下参数：

| 参数名                     | 类型   | 说明                                               |
|----------------------------|--------|----------------------------------------------------|
| `--path_ball_track_model`  | string | 预训练的球检测模型的路径                           |
| `--path_court_model`       | string | 预训练的场地检测模型的路径                         |
| `--path_bounce_model`      | string | 预训练的弹跳检测模型的路径                         |
| `--path_input_video`       | string | 输入视频文件的路径                                 |
| `--path_output_video`      | string | 输出视频文件的保存路径                             |

### 示例

```bash
python main.py \
    --path_ball_track_model models/ball_detector.pth \
    --path_court_model models/court_detector.pth \
    --path_bounce_model models/bounce_detector.pth \
    --path_input_video data/input_video.mp4 \
    --path_output_video results/output_video.mp4
```

---

## 6. 使用示例

假设你已经准备好了所有必要的模型文件，并且有一个输入视频 `tennis_match.mp4`，可以按照以下步骤运行主程序：

1. **准备模型**：

   确保以下模型文件存在：

   - `models/ball_detector.pth`
   - `models/court_detector.pth`
   - `models/bounce_detector.pth`

2. **运行程序**：

   ```bash
   python main.py \
       --path_ball_track_model models/ball_detector.pth \
       --path_court_model models/court_detector.pth \
       --path_bounce_model models/bounce_detector.pth \
       --path_input_video data/tennis_match.mp4 \
       --path_output_video results/tennis_match_output.mp4
   ```

3. **查看结果**：

   程序执行完成后，检查 `results/tennis_match_output.mp4`，其中应包含带有网球轨迹、弹跳点、球员检测框等可视化信息的视频。

---

## 7. 注意事项

- **模型兼容性**：确保使用的预训练模型与代码中的加载方式兼容。模型文件应与相应的检测模块匹配。
- **视频质量**：高分辨率和良好的光照条件有助于提高检测准确性。
- **计算资源**：若使用 GPU 进行加速，确保系统已正确配置 CUDA 环境。否则，程序将在 CPU 上运行，处理速度可能较慢。
- **依赖库版本**：确保安装的第三方库版本与代码要求兼容，以避免运行时错误。
- **错误处理**：程序未包含详细的错误处理机制，建议在实际应用中添加相应的异常处理代码。

---
