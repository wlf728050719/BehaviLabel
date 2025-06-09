import webbrowser
import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import os
from PIL import Image, ImageTk
import time


class BehaviLabel:
    def __init__(self, root,mode):
        self.root = root
        self.label_file = None
        self.labels = []
        self.video_dir = None
        self.video_index = 0
        self.video_list = []
        self.save_dir = None
        self.start_frame = None
        self.end_frame = None
        self.mode = mode
        self.cap = None
        self.paused = True
        self.allowed_speed = [1,2,3,4,8,16]
        self.speed_index = 0
        self.current_frame = 0
        self.total_frames = 0
        self.delay = 10
        self.video_width = 0
        self.video_height = 0
        self.current_photo = None  # 用于保持当前图像的引用
        self.selected_behavior = tk.StringVar()  # 存储选中的行为
        self.annotation_records = {}
        self.start_time = time.time()
        self.label_count = 0
        # 固定视频显示区域尺寸
        self.display_width = 1000  # 固定宽度
        self.display_height = 600  # 固定高度
        self.setup_ui()
        # 修改绑定方式，使用bind_all确保全局捕获空格键
        self.root.bind_all('<space>', self.pause_continue)
        self.root.bind_all('<a>', self.last_video)
        self.root.bind_all('<d>', self.next_video)
        self.root.bind_all('<w>', self.set_start_frame)
        self.root.bind_all('<s>', self.set_end_frame)
        self.root.bind_all('<Return>', self.confirm_annotation)
        self.root.bind_all('<Up>', self.select_prev_behavior)  # 添加上箭头绑定
        self.root.bind_all('<Down>', self.select_next_behavior)  # 添加下箭头绑定
        self.root.bind_all('<Left>', self.last_frame)
        self.root.bind_all('<Right>', self.next_frame)
        self.update()

    def update_working_time(self):
        self.lb_time.config(text=time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start_time)))
    def setup_ui(self):
        self.root.title("BehaviLabel")
        self.root.geometry("1500x800")

        filename_frame = tk.Frame(self.root)
        filename_frame.pack(side=tk.TOP, pady=5, fill=tk.X)

        self.lb_time = tk.Label(filename_frame, text="", font=("Arial", 10))
        self.lb_time.pack(side=tk.RIGHT, padx=5)

        self.lb_count = tk.Label(filename_frame, text="标记数目:(0)", font=("Arial", 10))
        self.lb_count.pack(side=tk.RIGHT, padx=5)

        self.filename_label = tk.Label(filename_frame, text="未选择文件", font=("Arial", 10), fg="blue")
        self.filename_label.pack(side=tk.RIGHT, padx=5)

        # 主frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧frame
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 右侧frame
        right_frame = tk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # 左上角按钮区
        button_frame = tk.Frame(left_frame)
        button_frame.pack(side=tk.TOP, anchor='nw', pady=5, padx=5, fill=tk.X)

        #初始化菜单
        self.btn_init = tk.Button(button_frame, text="初始化 ▼",
                                  command=lambda: self.show_menu(self.init_menu, self.btn_init))
        self.btn_init.pack(side=tk.LEFT, padx=5)
        self.init_menu = tk.Menu(self.root, tearoff=0)
        self.init_menu.add_command(label="设置视频目录", command=self.load_video_directory)
        self.init_menu.add_command(label="选择标签txt", command=self.load_label_file)
        self.init_menu.add_command(label="设置保存目录", command=self.load_save_dir)
        #基础操作菜单

        self.btn_base_use = tk.Button(button_frame, text="基础使用 ▼",
                                      command=lambda: self.show_menu(self.base_menu, self.btn_base_use))
        self.btn_base_use.pack(side=tk.LEFT, padx=5)
        self.base_menu = tk.Menu(self.root, tearoff=0)
        self.base_menu.add_command(label="上一个视频(A)",command=self.last_video)
        self.base_menu.add_command(label="下一个视频(D)", command=self.next_video)
        self.base_menu.add_separator()
        self.base_menu.add_command(label="快进(right)", command=self.next_frame)
        self.base_menu.add_command(label="后退(left)", command=self.last_frame)
        self.base_menu.add_separator()
        self.base_menu.add_command(label="设置起始帧(W)", command=self.set_start_frame)
        self.base_menu.add_command(label="设置结束帧(S)", command=self.set_end_frame)
        self.base_menu.add_separator()
        self.base_menu.add_command(label="切换上一行为类型(up)",command=self.select_prev_behavior)
        self.base_menu.add_command(label="切换下一行为类型(down)", command=self.select_next_behavior)
        self.base_menu.add_separator()
        self.base_menu.add_command(label="确认标注(Enter)",command=self.confirm_annotation)

        #设置菜单
        self.btn_setting = tk.Button(button_frame, text="设置 ▼",
                                     command=lambda: self.show_menu(self.setting_menu, self.btn_setting))
        self.btn_setting.pack(side=tk.LEFT, padx=5)
        self.setting_menu = tk.Menu(self.root, tearoff=0)
        self.setting_menu.add_command(label="连续播放", command=self.load_video_directory)

        # 关于菜单
        self.btn_util = tk.Button(button_frame, text="工具 ▼",
                                   command=lambda: self.show_menu(self.util_menu, self.btn_util))
        self.btn_util.pack(side=tk.LEFT, padx=5)
        self.util_menu = tk.Menu(self.root, tearoff=0)
        self.util_menu.add_command(label="视频分片", command=self.slice)
        self.util_menu.add_command(label="标记统计", command=self.show_statistics)


        #关于菜单
        self.btn_about = tk.Button(button_frame, text="关于 ▼",
                                     command=lambda: self.show_menu(self.about_menu, self.btn_about))
        self.btn_about.pack(side=tk.LEFT, padx=5)
        self.about_menu = tk.Menu(self.root, tearoff=0)
        self.about_menu.add_command(label="作者",command=self.author)
        self.about_menu.add_command(label="邮箱",command=self.mail)
        self.about_menu.add_command(label="项目地址",command=self.project)
        self.about_menu.add_command(label="检查更新",command=self.check_update)

        #倍速按钮
        self.btn_change_speed = tk.Button(button_frame, text="1倍速", command=self.change_speed)
        self.btn_change_speed.pack(side=tk.LEFT, padx=5)

        # 视频播放区域 - 固定大小的黑色背景
        self.video_canvas = tk.Canvas(left_frame,
                                      width=self.display_width,
                                      height=self.display_height,
                                      bg='black',
                                      highlightthickness=0)
        self.video_canvas.pack(side=tk.TOP, pady=10, padx=10)

        # 进度条
        progress_frame = tk.Frame(left_frame)
        progress_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        self.frame_label = tk.Label(progress_frame, text="0/0")
        self.frame_label.pack(side=tk.LEFT, padx=5)

        self.progress = ttk.Scale(progress_frame, from_=0, to=100, orient=tk.HORIZONTAL)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.progress.bind("<B1-Motion>", self.on_progress_drag)  # 拖动

        # 右上空白区域 - 现在添加帧信息和行为选择
        right_top_frame = tk.Frame(right_frame, bg='#f0f0f0')
        right_top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # 第一行：显示起始帧和结束帧
        frame_info_frame = tk.Frame(right_top_frame)
        frame_info_frame.pack(fill=tk.X, pady=5)

        tk.Label(frame_info_frame, text="起始帧:").pack(side=tk.LEFT)
        self.start_frame_label = tk.Label(frame_info_frame, text="未设置")
        self.start_frame_label.pack(side=tk.LEFT, padx=5)

        tk.Label(frame_info_frame, text="结束帧:").pack(side=tk.LEFT)
        self.end_frame_label = tk.Label(frame_info_frame, text="未设置")
        self.end_frame_label.pack(side=tk.LEFT, padx=5)

        # 第二行：行为选择下拉框
        behavior_frame = tk.Frame(right_top_frame)
        behavior_frame.pack(fill=tk.X, pady=5)

        tk.Label(behavior_frame, text="行为:").pack(side=tk.LEFT)
        self.behavior_combobox = ttk.Combobox(behavior_frame, textvariable=self.selected_behavior, state="readonly")
        self.behavior_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 第三行：确认按钮
        confirm_frame = tk.Frame(right_top_frame)
        confirm_frame.pack(fill=tk.X, pady=5)

        self.confirm_button = tk.Button(confirm_frame, text="确认标注", command=self.confirm_annotation)
        self.confirm_button.pack(fill=tk.X)

        # 右下区域 - 分成两个列表
        bottom_frame = tk.Frame(right_frame)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 视频列表框架
        video_list_frame = tk.Frame(bottom_frame)
        video_list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.lb_video_list = (tk.Label(video_list_frame, text="未设置视频目录"))
        self.lb_video_list.pack(side=tk.TOP)
        self.video_listbox = tk.Listbox(video_list_frame)
        self.video_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        video_scrollbar = tk.Scrollbar(video_list_frame, orient=tk.VERTICAL, command=self.video_listbox.yview)
        video_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.video_listbox.config(yscrollcommand=video_scrollbar.set)

        # 标注记录列表框架
        annotation_frame = tk.Frame(bottom_frame)
        annotation_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.lb_label_list = tk.Label(annotation_frame, text="未设置保存目录")
        self.lb_label_list.pack(side=tk.TOP)
        self.annotation_listbox = tk.Listbox(annotation_frame)
        self.annotation_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        annotation_scrollbar = tk.Scrollbar(annotation_frame, orient=tk.VERTICAL, command=self.annotation_listbox.yview)
        annotation_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.annotation_listbox.config(yscrollcommand=annotation_scrollbar.set)

        self.annotation_listbox.bind('<Button-3>', self.operate_record)  # 右键点击

        self.video_listbox.bind('<<ListboxSelect>>', self.on_video_select)
        self.video_listbox.bind('<FocusIn>', lambda e: self.root.focus_set())

    def operate_record(self, event):
        """右键点击标注记录时弹出提示框"""
        # 获取点击位置的索引
        index = self.annotation_listbox.nearest(event.y)
        if index < 0:
            return

        record = self.annotation_listbox.get(index)
        # 确保选中状态更新
        self.annotation_listbox.selection_clear(0, tk.END)
        self.annotation_listbox.selection_set(index)
        self.annotation_listbox.activate(index)

        # 创建弹出菜单
        popup = tk.Menu(self.root, tearoff=0)
        popup.add_command(label=f"记录详情: {record}")
        popup.add_separator()
        popup.add_command(label="定位记录", command=lambda: self.set_record_start_frame(record))
        popup.add_command(label="删除记录",command=lambda: self.delete_annotation_record(index))
        try:
            popup.tk_popup(event.x_root, event.y_root)
        finally:
            popup.grab_release()

    def set_record_start_frame(self, record):
        frame_range, behavior = record.split(": ")
        start_frame, end_frame = map(int, frame_range.split("-"))
        self.current_frame = start_frame
        self.paused = True
        self.show_frame(self.video_list[self.video_index])
        self.update_progress()

    def delete_annotation_record(self, index):
        """从TXT文件中删除指定的标注记录"""
        if not self.video_list or self.video_index >= len(self.video_list):
            return

        # 获取当前视频文件名(不带扩展名)
        video_path = self.video_list[self.video_index]
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # 确保保存目录已设置
        if not self.save_dir:
            return

        # 构建标注文件路径
        record_file = os.path.join(self.save_dir, f"{video_name}.txt")

        # 获取要删除的记录内容
        record_to_delete = self.annotation_listbox.get(index)

        try:
            # 读取所有记录
            with open(record_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 过滤掉要删除的记录
            new_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 3:
                        start_frame = parts[0]
                        end_frame = parts[1]
                        behavior = ' '.join(parts[2:])
                        current_record = f"{start_frame}-{end_frame}: {behavior}"
                        if current_record != record_to_delete:
                            new_lines.append(line + '\n')

            # 重新写入文件
            with open(record_file, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

            # 更新界面显示
            self.load_records()

            print(f"已删除记录: {record_to_delete}")
        except Exception as e:
            self.show_custom_message(f"删除记录失败: {str(e)}")
        self.load_records()

    def show_menu(self, menu, button):
        """通用显示菜单方法"""
        try:
            menu.tk_popup(button.winfo_rootx(),
                          button.winfo_rooty() + button.winfo_height())
        finally:
            menu.grab_release()
    def update(self):
        if len(self.video_list) > 0 and not self.paused:
            self.show_frame(self.video_list[self.video_index])
            self.update_progress()
        self.update_working_time()
        self.root.after(self.delay, self.update)

    def load_records(self):
        """从当前视频文件对应的标注txt中加载标注记录"""
        # 确保有视频文件被选中
        if not self.video_list or self.video_index >= len(self.video_list):
            return

        # 获取当前视频文件名(不带扩展名)
        video_path = self.video_list[self.video_index]
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # 确保保存目录已设置
        if not self.save_dir:
            return

        # 构建标注文件路径
        record_file = os.path.join(self.save_dir, f"{video_name}.txt")

        # 清空当前记录
        self.annotation_listbox.delete(0, tk.END)
        self.annotation_records[video_name] = []

        # 检查标注文件是否存在
        if os.path.exists(record_file):
            try:
                with open(record_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            # 解析标注记录 (格式: 起始帧 结束帧 行为)
                            parts = line.split()
                            if len(parts) >= 3:
                                start_frame = parts[0]
                                end_frame = parts[1]
                                behavior = ' '.join(parts[2:])  # 处理行为名称中可能包含空格的情况
                                record_str = f"{start_frame}-{end_frame}: {behavior}"
                                self.annotation_records[video_name].append(record_str)
                                self.annotation_listbox.insert(tk.END, record_str)
            except Exception as e:
                self.show_custom_message(f"加载标注记录失败: {str(e)}")

    def change_speed(self):
        self.speed_index+=1
        if self.speed_index>=len(self.allowed_speed):
            self.speed_index=0
        self.info("current speed:" + str(self.allowed_speed[self.speed_index]))
        self.btn_change_speed.config(text=str(self.allowed_speed[self.speed_index])+'倍速')

    def on_progress_drag(self, event):
        """拖动进度条时实时输出当前值（带防抖）"""
        self.paused = True
        if not hasattr(self, 'last_drag') or time.time() - self.last_drag > 0.1:  # 0.1秒防抖
            self.last_drag = time.time()
            # 计算点击位置对应的帧数
            if self.total_frames > 0:
                # 获取进度条宽度
                width = self.progress.winfo_width()
                # 计算点击位置百分比
                click_pos = event.x / width
                # 计算对应的帧数
                new_frame = int(click_pos * self.total_frames)
                # 确保帧数在有效范围内
                new_frame = max(0, min(new_frame, self.total_frames - 1))
                # 更新当前帧
                self.current_frame = new_frame
                # 更新显示
                self.show_frame(self.video_list[self.video_index])
                self.frame_label.config(text=f"{self.current_frame}/{self.total_frames}")

    def on_video_select(self, event):
        selection = self.video_listbox.curselection()
        if selection:
            index = selection[0]
            # 如果切换的是不同的视频才重置current_frame
            if index != self.video_index:
                self.current_frame = 0
            filepath = self.video_list[index]  # 取完整路径
            total = len(self.video_list)
            abs_path = os.path.abspath(filepath)
            self.filename_label.config(
                text=f"{abs_path}（{index + 1}/{total}）"
            )
            self.video_index = index
            self.root.focus_set()
            self.paused = True
            self.show_frame(self.video_list[self.video_index])
            self.update_progress()
            self.load_records()

    def confirm_annotation(self, event=None):
        """确认标注按钮的回调函数"""
        behavior = self.selected_behavior.get()
        if not behavior:
            self.show_custom_message("请先选择一个行为")
            return

        if self.start_frame is None or self.end_frame is None:
            self.show_custom_message("请先设置起始帧和结束帧")
            return

        # 确保保存目录已设置
        if not self.save_dir:
            self.show_custom_message("请先设置保存目录")
            return

        # 确保有视频文件被选中
        if not self.video_list or self.video_index >= len(self.video_list):
            self.show_custom_message("没有视频文件被选中")
            return

        # 获取当前视频文件名(不带扩展名)
        video_path = self.video_list[self.video_index]
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # 构建保存路径
        save_path = os.path.join(self.save_dir, f"{video_name}.txt")

        try:
            # 写入标注信息(追加模式)
            with open(save_path, 'a', encoding='utf-8') as f:
                f.write(f"{self.start_frame} {self.end_frame} {behavior}\n")
        except Exception as e:
            self.show_custom_message(f"保存标注失败: {str(e)}")
            return

        # 重置帧标记
        self.start_frame = None
        self.end_frame = None
        self.start_frame_label.config(text="未设置")
        self.end_frame_label.config(text="未设置")
        self.load_records()

        self.label_count+=1
        self.lb_count.config(text="标记数目("+str(self.label_count)+")")

    def last_frame(self,event=None):
        self.paused = True
        self.current_frame -= self.allowed_speed[self.speed_index]
        if self.current_frame < 0:
            self.current_frame = 0
        self.show_frame(self.video_list[self.video_index])
        self.update_progress()

    def next_frame(self,event=None):
        self.paused = True
        self.current_frame += self.allowed_speed[self.speed_index]
        if self.current_frame >= self.total_frames:
            self.current_frame = self.total_frames-1
        self.show_frame(self.video_list[self.video_index])
        self.update_progress()

    def select_prev_behavior(self, event=None):
        """选择上一个行为"""
        if not self.labels:
            return

        current = self.selected_behavior.get()
        if current in self.labels:
            index = self.labels.index(current)
            if index > 0:
                self.selected_behavior.set(self.labels[index - 1])
        elif self.labels:
            self.selected_behavior.set(self.labels[-1])
        return "break"  # 阻止事件继续传播

    def select_next_behavior(self, event=None):
        """选择下一个行为"""
        if not self.labels:
            return

        current = self.selected_behavior.get()
        if current in self.labels:
            index = self.labels.index(current)
            if index < len(self.labels) - 1:
                self.selected_behavior.set(self.labels[index + 1])
        elif self.labels:
            self.selected_behavior.set(self.labels[0])
        return "break"  # 阻止事件继续传播

    def pause_continue(self, event=None):
        self.paused = not self.paused
        return "break"  # 阻止事件继续传播

    def next_video(self,event=None):
        if self.video_index < len(self.video_list) - 1:
            self.video_index += 1
            filepath = self.video_list[self.video_index]  # 取完整路径
            total = len(self.video_list)
            abs_path = os.path.abspath(filepath)
            self.filename_label.config(
                text=f"{abs_path}（{self.video_index + 1}/{total}）"
            )
            self.current_frame = 0
            self.root.focus_set()
            self.paused = True
            self.show_frame(self.video_list[self.video_index])
            self.update_progress()
            self.load_records()
        else:
            msg = f"已经是最后一个视频"
            self.show_custom_message(msg)

    def last_video(self,event=None):
        if self.video_index >= 1:
            self.video_index -= 1
            filepath = self.video_list[self.video_index]  # 取完整路径
            total = len(self.video_list)
            abs_path = os.path.abspath(filepath)
            self.filename_label.config(
                text=f"{abs_path}（{self.video_index + 1}/{total}）"
            )
            self.current_frame = 0
            self.root.focus_set()
            self.paused = True
            self.show_frame(self.video_list[self.video_index])
            self.update_progress()
            self.load_records()
        else:
            msg = f"已经是第一个视频"
            self.show_custom_message(msg)

    def set_start_frame(self,event=None):
        self.start_frame = self.current_frame
        self.start_frame_label.config(text=str(self.start_frame))
        self.paused = True
        self.info('set start frame:' + str(self.start_frame))

    def set_end_frame(self,event=None):
        self.end_frame = self.current_frame
        self.end_frame_label.config(text=str(self.end_frame))
        self.paused = True
        self.info('set end frame:' + str(self.end_frame))

    def load_label_file(self):
        """加载标签文件"""
        file_path = filedialog.askopenfilename(title="选择标签文件", filetypes=[("文本文件", "*.txt")])
        if file_path:
            self.label_file = file_path
            with open(file_path, 'r', encoding='utf-8') as f:
                self.labels = [line.strip() for line in f.readlines() if line.strip()]
            # 更新下拉框选项
            self.behavior_combobox['values'] = self.labels
            if self.labels:
                self.selected_behavior.set(self.labels[0])
            msg = f"已加载 {len(self.labels)} 个行为标签"
            self.show_custom_message(msg)

    def load_save_dir(self):
        """设置保存目录"""
        dir_path = filedialog.askdirectory(title="选择保存目录")
        if dir_path:
            self.save_dir = dir_path
            msg = f"标注文件将保存到: {dir_path}"
            self.show_custom_message(msg)
            self.lb_label_list.config(text='标注记录')

    def load_video_directory(self):
        """设置视频目录"""
        directory = filedialog.askdirectory()
        if directory:
            self.video_dir = directory
            self.video_list.clear()
            self.video_listbox.delete(0, tk.END)
            for filename in os.listdir(directory):
                if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    path = os.path.join(directory, filename)
                    self.video_list.append(path)
                    self.video_listbox.insert(tk.END, filename)  # 只插入文件名
            msg = f"找到 {len(self.video_list)} 个视频文件"
            self.show_custom_message(msg)
            self.lb_video_list.config(text="视频目录")

    def show_custom_message(self, message, links=None):
        """显示自定义消息框，支持超链接和文本复制"""
        top = tk.Toplevel(self.root)
        top.title("提示")
        top.resizable(False, False)

        # 使用Text控件实现可复制文本和超链接
        text = tk.Text(top, wrap=tk.WORD, height=10, width=50,
                       padx=10, pady=10, font=('Arial', 10))
        text.pack()

        # 解析消息文本
        for line in message.split('\n'):
            # 查找行中是否包含链接
            url_found = False
            if links:
                for url in links:
                    if url in line:
                        # 为每个链接创建唯一tag
                        tag_name = f"hyperlink_{url}"

                        # 配置当前链接样式
                        text.tag_config(tag_name, foreground="blue", underline=1)
                        text.tag_bind(tag_name, "<Enter>",
                                      lambda e, t=text: t.config(cursor="hand2"))
                        text.tag_bind(tag_name, "<Leave>",
                                      lambda e, t=text: t.config(cursor=""))

                        # 分割普通文本和URL
                        parts = line.split(url)
                        text.insert(tk.END, parts[0])
                        text.insert(tk.END, url, tag_name)
                        if len(parts) > 1:
                            text.insert(tk.END, parts[1])
                        text.insert(tk.END, "\n")

                        # 绑定点击事件（使用默认参数捕获当前url值）
                        text.tag_bind(tag_name, "<Button-1>",
                                      lambda e, u=url: webbrowser.open(links[u]))
                        url_found = True
                        break

            if not url_found:
                text.insert(tk.END, line + "\n")

        # 使文本只读但可选择复制
        text.config(state=tk.DISABLED)

        # 确定按钮
        btn = tk.Button(top, text="确定", command=top.destroy)
        btn.pack(pady=5)

        # 窗口居中
        top.update_idletasks()
        width = top.winfo_width()
        height = top.winfo_height()
        x = (top.winfo_screenwidth() // 2) - (width // 2)
        y = (top.winfo_screenheight() // 2) - (height // 2)
        top.geometry(f'+{x}+{y}')

    def show_frame(self, video_path):
        self.debug(self.current_frame)
        # 释放之前的资源
        if self.cap is not None:
            self.cap.release()

        # 清除画布上的内容
        self.video_canvas.delete("all")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            return

        # 获取视频原始尺寸
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            # 转换颜色空间
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)

            # 计算保持比例的缩放因子
            ratio = min(self.display_width / self.video_width,
                        self.display_height / self.video_height)
            new_width = int(self.video_width * ratio)
            new_height = int(self.video_height * ratio)

            # 调整图像大小
            img = img.resize((new_width, new_height), Image.LANCZOS)

            # 计算居中位置
            x_offset = (self.display_width - new_width) // 2
            y_offset = (self.display_height - new_height) // 2

            # 创建并显示图像
            self.current_photo = ImageTk.PhotoImage(image=img)
            self.video_canvas.create_image(x_offset, y_offset,
                                           anchor=tk.NW,
                                           image=self.current_photo)

        # 只有在播放状态下才前进到下一帧
        if not self.paused:
            self.current_frame += self.allowed_speed[self.speed_index]

        # 确保current_frame不超过视频总帧数
        if self.current_frame >= self.total_frames:
            self.current_frame = self.total_frames - 1

    def update_progress(self):
        """更新进度条和帧数显示"""
        if self.total_frames > 0:
            progress_value = (self.current_frame / self.total_frames) * 100
            self.progress.set(progress_value)
            self.frame_label.config(text=f"{self.current_frame}/{self.total_frames}")

    def author(self):
        """显示作者信息，带可点击链接"""
        msg = (f"name: 汪洛飞(Luofei Wang)\n"
               f"blog: https://blog.csdn.net/wlf2030\n"
               f"github: https://github.com/wlf728050719\n")

        links = {
            "https://blog.csdn.net/wlf2030": "https://blog.csdn.net/wlf2030",
            "https://github.com/wlf728050719": "https://github.com/wlf728050719"
        }

        self.show_custom_message(msg, links)

    def mail(self):
        """显示邮箱，可点击发送邮件"""
        email = "18086270070@163.com"
        msg = f"邮箱: {email}"

        # 创建mailto链接
        mailto = f"mailto:{email}"
        links = {email: mailto}

        self.show_custom_message(msg, links)

    def project(self):
        """显示项目链接，可点击打开"""
        url = "https://github.com/wlf728050719/BehaviLabel"
        msg = f"项目地址: {url}"

        links = {url: url}
        self.show_custom_message(msg, links)

    def check_update(self):
        """检查更新"""
        # 这里可以添加实际的更新检查逻辑
        self.show_custom_message("正在检查更新...\n暂未实现自动更新功能")

    def show_statistics(self):
        """统计标记信息功能"""
        # 创建统计窗口
        stat_window = tk.Toplevel(self.root)
        stat_window.title("标记统计")
        stat_window.attributes('-topmost', True)
        stat_window.grab_set()

        # 主框架
        main_frame = tk.Frame(stat_window, padx=10, pady=10)
        main_frame.pack()

        # 选择TXT目录
        txt_dir_var = tk.StringVar(value=self.save_dir)

        def select_txt_dir():
            stat_window.attributes('-topmost', False)
            dir_path = filedialog.askdirectory(
                title="选择TXT目录",
                initialdir=self.save_dir,
                parent=stat_window
            )
            stat_window.attributes('-topmost', True)
            if dir_path:
                txt_dir_var.set(dir_path)

        dir_frame = tk.Frame(main_frame)
        dir_frame.pack(fill=tk.X, pady=5)
        tk.Label(dir_frame, text="TXT目录:").pack(side=tk.LEFT)
        tk.Entry(dir_frame, textvariable=txt_dir_var, width=40).pack(side=tk.LEFT, padx=5)
        tk.Button(dir_frame, text="浏览...", command=select_txt_dir).pack(side=tk.LEFT)

        # 进度条
        progress_var = tk.DoubleVar()
        progress_var.set(0)
        progress_frame = tk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=10)
        tk.Label(progress_frame, text="进度:").pack(side=tk.LEFT)
        progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=100)
        progress_bar.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        # 状态标签
        status_label = tk.Label(main_frame, text="准备统计...", fg="blue")
        status_label.pack()

        # 结果文本框
        result_text = tk.Text(main_frame, wrap=tk.WORD, height=15, width=60,
                              padx=5, pady=5, font=('Consolas', 10))
        result_text.pack(pady=5)

        # 开始统计按钮
        def start_statistics():
            txt_dir = txt_dir_var.get()
            if not txt_dir:
                self.show_custom_message("请选择TXT目录")
                return

            # 禁用按钮
            stat_btn.config(state=tk.DISABLED)

            try:
                # 执行统计
                total_marks = 0
                total_frames = 0
                action_stats = {}

                # 获取所有txt文件
                txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]
                total_files = len(txt_files)

                for i, txt_file in enumerate(txt_files):
                    # 更新进度
                    progress = (i + 1) / total_files * 100
                    progress_var.set(progress)
                    status_label.config(text=f"正在统计 {txt_file} ({i + 1}/{total_files})")
                    stat_window.update_idletasks()

                    # 读取txt文件内容
                    txt_path = os.path.join(txt_dir, txt_file)
                    with open(txt_path, 'r') as f:
                        lines = f.readlines()

                    # 统计每行标记
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 3:
                            continue

                        start_frame = int(parts[0])
                        end_frame = int(parts[1])
                        action = parts[2]

                        # 统计总数
                        total_marks += 1
                        total_frames += (end_frame - start_frame + 1)

                        # 统计行为
                        if action not in action_stats:
                            action_stats[action] = {
                                'count': 0,
                                'frames': 0
                            }
                        action_stats[action]['count'] += 1
                        action_stats[action]['frames'] += (end_frame - start_frame + 1)

                # 显示统计结果
                result_text.config(state=tk.NORMAL)
                result_text.delete(1.0, tk.END)

                result_text.insert(tk.END, f"=== 标记统计结果 ===\n\n")
                result_text.insert(tk.END, f"总标记数: {total_marks}\n")
                result_text.insert(tk.END, f"总帧数: {total_frames}\n\n")

                result_text.insert(tk.END, f"=== 按行为统计 ===\n")
                for action, stats in sorted(action_stats.items()):
                    result_text.insert(tk.END,
                                       f"{action}: {stats['count']} 条, {stats['frames']} 帧\n")

                result_text.config(state=tk.DISABLED)
                status_label.config(text="统计完成", fg="green")

            except Exception as e:
                self.show_custom_message(f"统计出错: {str(e)}")
            finally:
                stat_btn.config(state=tk.NORMAL)

        stat_btn = tk.Button(main_frame, text="开始统计", command=start_statistics)
        stat_btn.pack(pady=10)

        # 窗口关闭处理
        def on_closing():
            stat_window.grab_release()
            stat_window.destroy()

        stat_window.protocol("WM_DELETE_WINDOW", on_closing)

        # 窗口居中
        stat_window.update_idletasks()
        width = stat_window.winfo_width()
        height = stat_window.winfo_height()
        x = (stat_window.winfo_screenwidth() // 2) - (width // 2)
        y = (stat_window.winfo_screenheight() // 2) - (height // 2)
        stat_window.geometry(f'+{x}+{y}')

    def slice(self):
        """视频分片功能主方法"""
        # 创建选择窗口并设置为顶级窗口
        top = tk.Toplevel(self.root)
        top.title("视频分片设置")
        top.resizable(False, False)
        top.attributes('-topmost', True)  # 设置为最顶层
        top.grab_set()  # 独占焦点

        # 存储选择的路径
        selected_paths = {
            'video_dir': tk.StringVar(value=self.video_dir),
            'txt_dir': tk.StringVar(value=self.save_dir),
            'output_dir': tk.StringVar()
        }

        # 创建进度条变量
        progress_var = tk.DoubleVar()
        progress_var.set(0)

        # 创建主框架
        main_frame = tk.Frame(top, padx=10, pady=10)
        main_frame.pack()

        # 视频目录选择
        def select_video_dir():
            top.attributes('-topmost', False)  # 临时取消最顶层属性
            dir_path = filedialog.askdirectory(
                title="选择视频目录",
                initialdir=self.video_dir,
                parent=top  # 指定父窗口
            )
            top.attributes('-topmost', True)  # 恢复最顶层属性
            if dir_path:
                selected_paths['video_dir'].set(dir_path)

        video_frame = tk.Frame(main_frame)
        video_frame.pack(fill=tk.X, pady=5)
        tk.Label(video_frame, text="视频目录:").pack(side=tk.LEFT)
        tk.Entry(video_frame, textvariable=selected_paths['video_dir'], width=40).pack(side=tk.LEFT, padx=5)
        tk.Button(video_frame, text="浏览...", command=select_video_dir).pack(side=tk.LEFT)

        # TXT目录选择
        def select_txt_dir():
            top.attributes('-topmost', False)
            dir_path = filedialog.askdirectory(
                title="选择TXT目录",
                initialdir=self.save_dir,
                parent=top
            )
            top.attributes('-topmost', True)
            if dir_path:
                selected_paths['txt_dir'].set(dir_path)

        txt_frame = tk.Frame(main_frame)
        txt_frame.pack(fill=tk.X, pady=5)
        tk.Label(txt_frame, text="TXT目录:").pack(side=tk.LEFT)
        tk.Entry(txt_frame, textvariable=selected_paths['txt_dir'], width=40).pack(side=tk.LEFT, padx=5)
        tk.Button(txt_frame, text="浏览...", command=select_txt_dir).pack(side=tk.LEFT)

        # 输出目录选择
        def select_output_dir():
            top.attributes('-topmost', False)
            dir_path = filedialog.askdirectory(
                title="选择输出目录",
                parent=top
            )
            top.attributes('-topmost', True)
            if dir_path:
                selected_paths['output_dir'].set(dir_path)

        output_frame = tk.Frame(main_frame)
        output_frame.pack(fill=tk.X, pady=5)
        tk.Label(output_frame, text="输出目录:").pack(side=tk.LEFT)
        tk.Entry(output_frame, textvariable=selected_paths['output_dir'], width=40).pack(side=tk.LEFT, padx=5)
        tk.Button(output_frame, text="浏览...", command=select_output_dir).pack(side=tk.LEFT)

        # 进度条
        progress_frame = tk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=10)
        tk.Label(progress_frame, text="进度:").pack(side=tk.LEFT)
        progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=100)
        progress_bar.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        # 状态标签
        status_label = tk.Label(main_frame, text="", fg="blue")
        status_label.pack()

        # 确认按钮
        def start_processing():
            # 验证输入
            if not selected_paths['output_dir'].get():
                self.show_custom_message("请选择输出目录")
                return

            # 禁用按钮
            confirm_btn.config(state=tk.DISABLED)

            # 开始处理
            try:
                success = self._process_videos(
                    video_dir=selected_paths['video_dir'].get(),
                    txt_dir=selected_paths['txt_dir'].get(),
                    output_dir=selected_paths['output_dir'].get(),
                    progress_var=progress_var,
                    status_label=status_label,
                    top_window=top
                )

                # 处理完成后关闭进度窗口
                top.grab_release()
                top.destroy()

                # 显示完成消息（会自动置顶）
                self.show_custom_message("视频分片完成！")

            except Exception as e:
                # 出错时也关闭进度窗口
                top.grab_release()
                top.destroy()
                self.show_custom_message(f"处理出错: {str(e)}")

        confirm_btn = tk.Button(main_frame, text="开始分片", command=start_processing)
        confirm_btn.pack(pady=10)

        # 窗口关闭时的处理
        def on_closing():
            top.grab_release()
            top.destroy()

        top.protocol("WM_DELETE_WINDOW", on_closing)

        # 窗口居中
        top.update_idletasks()
        width = top.winfo_width()
        height = top.winfo_height()
        x = (top.winfo_screenwidth() // 2) - (width // 2)
        y = (top.winfo_screenheight() // 2) - (height // 2)
        top.geometry(f'+{x}+{y}')

    def _process_videos(self, video_dir, txt_dir, output_dir, progress_var, status_label, top_window):
        """实际处理视频的方法"""
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 获取所有txt文件
        txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]
        total_files = len(txt_files)

        for i, txt_file in enumerate(txt_files):
            # 更新进度和状态
            progress = (i + 1) / total_files * 100
            progress_var.set(progress)
            status_label.config(text=f"正在处理 {txt_file} ({i + 1}/{total_files})")
            top_window.update_idletasks()  # 使用传入的窗口对象更新UI

            # 获取对应的视频文件路径
            video_name = os.path.splitext(txt_file)[0]
            video_path = os.path.join(video_dir, video_name)

            # 检查是否有对应的视频文件（支持多种视频格式）
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            found_video = False
            for ext in video_extensions:
                if os.path.exists(video_path + ext):
                    video_path += ext
                    found_video = True
                    break

            if not found_video:
                continue

            # 读取视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 读取txt文件内容
            txt_path = os.path.join(txt_dir, txt_file)
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            # 处理每一行标记
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                start_frame = int(parts[0])
                end_frame = int(parts[1])
                action = parts[2]

                # 确保行为文件夹存在
                action_folder = os.path.join(output_dir, action)
                if not os.path.exists(action_folder):
                    os.makedirs(action_folder)

                # 创建输出视频文件名
                output_name = f"{video_name}_{start_frame}_{end_frame}_{action}.mp4"
                output_path = os.path.join(action_folder, output_name)

                # 设置视频写入器
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

                # 跳转到起始帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                # 读取并写入指定范围内的帧
                for frame_num in range(start_frame, end_frame + 1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)

                out.release()

            cap.release()

        return True  # 返回成功状态

    def debug(self, string):
        if self.mode == 'debug':
            print(string)

    def info(self, string):
        if self.mode == 'info' or self.mode == 'debug':
            print(string)

if __name__ == "__main__":
    root = tk.Tk()
    app = BehaviLabel(root, 'debug')
    root.mainloop()