# BehaviLabel  
[Luofei Wang](https://github.com/wlf728050719)

*A script for fast video behavior labeling (快速视频行为标注工具)*

## Quick Start  

**0.** Run the packaged EXE file or execute `python main.py` in a Python environment  
*(点击[此处](https://github.com/wlf728050719/BehaviLabel/releases/download/v1.0.0/dist.zip)下载解压并运行打包好的exe文件*，或在Python环境下执行`python main.py`)

**1.** Click the "Initialize" button to set:  
- Root directory of videos to be labeled (待标注视频根目录)  
- Behavior type TXT file (行为类型txt文件)  
- Label file storage directory (标记文件存放目录)  

After setup, the right panel will display all video files and their existing labels.  
*(设置后程序右侧列表会显示设置目录的所有视频文件及标注记录)*  

![Demo Image](images/quickstart01.png)  

**2.** Use the control buttons to:  
- Navigate between videos (视频切换)  
- Fast forward/rewind (快进/后退)  
- Switch label types (切换标注类型)  
- Set start/end frames and confirm labels (设置标注起始帧并确认)  
- Use progress bar for quick positioning (进度条快速定位)  

![Demo Image](images/quickstart02.png)  

**3.** Right-click on existing labels to:  
- Jump to the labeled frame (快速定位到标注帧)  
- Delete the label (删除标注)  

![Demo Image](images/quickstart03.png)  

**4.** Additional features:  
- Multiple playback speed options (多种倍速调节)  
- Work duration timer (工作时长统计)  
- Label count display (标注记录统计)  

## Utilities  

**1. Label Statistics**  
View overall labeling statistics (标注整体情况统计)  

![Demo Image](images/util1.png)  

**2. Video Clipping**  
Extract video segments based on labels (根据标注记录切片视频):  
- Segments with same behavior are saved in same subfolder (相同行为存放在同一文件夹)  
- Naming format: `[original_name]_[start_frame]_[end_frame]_[behavior].mp4`  

![Demo Image](images/util2.png)  

## Future Features  

1. History records & language switching (历史记录与语言切换功能)  
2. Direct skeleton node CSV generation (直接生成骨骼节点csv文件)  
