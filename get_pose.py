import time
import numpy as np
import cv2
import onnxruntime
import warnings
import argparse
import json
import os

warnings.filterwarnings("ignore")

# 颜色定义
palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])

# 骨架连接定义 (COCO格式)
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True):
    ''' 调整图像大小和两边灰条填充 '''
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im


def xyxy2xywh(x):
    y = np.copy(x)
    y[:, 2] = x[:, 2] - x[:, 0]  # w
    y[:, 3] = x[:, 3] - x[:, 1]  # h
    return y


def scale_boxes(img1_shape, boxes, img0_shape):
    ''' 将预测的坐标信息转换回原图尺度 '''
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    boxes[:, 0] -= pad[0]
    boxes[:, 1] -= pad[1]
    boxes[:, :4] /= gain
    num_kpts = boxes.shape[1] // 3
    for kid in range(2, num_kpts):
        boxes[:, kid * 3] = (boxes[:, kid * 3] - pad[0]) / gain
        boxes[:, kid * 3 + 1] = (boxes[:, kid * 3 + 1] - pad[1]) / gain
    clip_boxes(boxes, img0_shape)
    return boxes


def clip_boxes(boxes, shape):
    top_left_x = boxes[:, 0].clip(0, shape[1])
    top_left_y = boxes[:, 1].clip(0, shape[0])
    bottom_right_x = (boxes[:, 0] + boxes[:, 2]).clip(0, shape[1])
    bottom_right_y = (boxes[:, 1] + boxes[:, 3]).clip(0, shape[0])
    boxes[:, 0] = top_left_x
    boxes[:, 1] = top_left_y
    boxes[:, 2] = bottom_right_x
    boxes[:, 3] = bottom_right_y


def read_img(img, img_mean=127.5, img_scale=1 / 127.5):
    img = (img - img_mean) * img_scale
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img, 0)
    img = img.transpose(0, 3, 1, 2)
    return img


def plot_skeleton_kpts(im, kpts, steps=3):
    ''' 在图像上绘制关键点和骨架 '''
    num_kpts = len(kpts) // steps
    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        conf = kpts[steps * kid + 2]
        if conf > 0.5:
            cv2.circle(im, (int(x_coord), int(y_coord)), 3, (int(r), int(g), int(b)), -1)
    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0] - 1) * steps]), int(kpts[(sk[0] - 1) * steps + 1]))
        pos2 = (int(kpts[(sk[1] - 1) * steps]), int(kpts[(sk[1] - 1) * steps + 1]))
        conf1 = kpts[(sk[0] - 1) * steps + 2]
        conf2 = kpts[(sk[1] - 1) * steps + 2]
        if conf1 > 0.5 and conf2 > 0.5:
            cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)


def initialize_session(model_path):
    ''' 初始化ONNX Runtime会话 '''
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return None

    try:
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        ort_session = onnxruntime.InferenceSession(model_path,
                                                   session_options=session_options,
                                                   providers=['CPUExecutionProvider'])
        print(f"成功加载模型: {model_path}")
        return ort_session
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None


def process_frame(ort_session, img, conf_threshold=0.1):
    ''' 处理单帧图像 '''
    # 图像预处理
    image1 = letterbox(img)
    input = read_img(image1, 0.0, 0.00392156862745098)
    input_name = ort_session.get_inputs()[0].name

    # 模型推理
    output = ort_session.run([], {input_name: input})[0]

    # 置信度过滤
    output = output[output[..., 4] > conf_threshold]
    if len(output) == 0:
        return img, None  # 没有检测到任何目标，返回原图和空关键点

    # 坐标转换
    det_box = xyxy2xywh(output)
    output = scale_boxes(image1.shape, det_box, img.shape)

    # 提取检测框和关键点
    det_bboxes, det_scores, det_labels, kpts = output[:, 0:4], output[:, 4], output[:, 5], output[:, 6:]

    # 绘制骨骼关键点
    for idx in range(len(det_bboxes)):
        kpt = kpts[idx]
        plot_skeleton_kpts(img, kpt)

    return img, kpts[0]  # 返回处理后的图像和第一个人的关键点




