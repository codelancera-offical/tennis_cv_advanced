以下是使用 FFmpeg 进行视频分辨率调整和格式转换的最简说明：

### 1. **调整视频分辨率**
将视频调整为 1280x720 分辨率（720p）：

```bash
ffmpeg -i input_video.mp4 -vf scale=1280:720 output_video.mp4
```

- `input_video.mp4`：输入文件。
- `scale=1280:720`：指定输出分辨率为 1280x720。
- `output_video.mp4`：输出文件。

### 2. **格式转换**
将视频从一种格式转换为另一种格式（如从 MP4 转换为 AVI）：

```bash
ffmpeg -i input_video.mp4 output_video.avi
```

- `input_video.mp4`：输入文件。
- `output_video.avi`：输出文件，扩展名决定输出格式。

### 3. **调整分辨率并转换格式**
同时调整视频分辨率并转换格式，例如从 MP4 转换为 MKV 并将分辨率设为 1280x720：

```bash
ffmpeg -i input_video.mp4 -vf scale=1280:720 output_video.mkv
```

### 4. **保持纵横比**
在调整分辨率时自动保持原视频的纵横比：

```bash
ffmpeg -i input_video.mp4 -vf scale=1280:-1 output_video.mp4
```

- `scale=1280:-1`：宽度设为 1280，高度根据原视频纵横比自动调整。

---

这就是 FFmpeg 最简明的调整分辨率和格式转换的使用说明，适合处理大多数常见视频文件。