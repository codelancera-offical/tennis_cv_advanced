要将视频的分辨率调整到 1280x720（即 720p 分辨率），可以使用 FFmpeg 这一常用的命令行工具进行操作。FFmpeg 是一个强大的多媒体处理工具，支持视频格式转换、视频编辑、压缩、调整分辨率等操作。

### 使用 FFmpeg 调整视频分辨率到 1280x720 的命令：
```bash
ffmpeg -i input_video.mp4 -vf scale=1280:720 output_video.mp4
```

### 参数解释：
- `-i input_video.mp4`：这是输入视频文件的路径。
- `-vf scale=1280:720`：`-vf` 是 FFmpeg 的视频滤镜选项，`scale=1280:720` 指定了输出视频的分辨率。
- `output_video.mp4`：这是输出视频文件的路径。

### 附加设置：
你可以根据需要添加更多的选项来优化输出视频的质量和文件大小。

1. **设置输出视频的比特率（控制质量）**：
   如果想控制输出视频的比特率以保证视频质量，可以添加 `-b:v` 选项。例如，设置比特率为 2 Mbps：
   ```bash
   ffmpeg -i input_video.mp4 -vf scale=1280:720 -b:v 2M output_video.mp4
   ```

2. **保持纵横比**：
   如果你想确保视频调整分辨率时保持原始视频的纵横比，可以使用 `-vf scale=1280:-1`（`-1` 表示保持纵横比，自动调整高度）：
   ```bash
   ffmpeg -i input_video.mp4 -vf scale=1280:-1 output_video.mp4
   ```

3. **更改视频格式**：
   如果你想输出不同的视频格式，可以通过更改输出文件的扩展名实现，例如 `.avi`、`.mkv`、`.mov` 等：
   ```bash
   ffmpeg -i input_video.mp4 -vf scale=1280:720 output_video.mkv
   ```

### 安装 FFmpeg：
如果你还没有安装 FFmpeg，可以在不同系统下安装：
- **Windows**：可以从 [FFmpeg官网](https://ffmpeg.org/download.html) 下载预编译的二进制文件，解压后将其添加到系统的环境变量中。
- **Linux**：通过包管理器安装，例如：
  ```bash
  sudo apt-get install ffmpeg  # Ubuntu/Debian
  sudo yum install ffmpeg      # CentOS/RHEL
  ```
- **macOS**：通过 Homebrew 安装：
  ```bash
  brew install ffmpeg
  ```

这样你就可以将视频的分辨率调整为 1280x720，并输出到指定的视频文件中。