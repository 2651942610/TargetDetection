import argparse
import os
import platform
import sys
from pathlib import Path
import torch
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.myutil import Globals
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights='yolov5s.pt',  # 模型路径或triton URL
        source='data/images',  # 文件/目录/URL/glob/screen/0(摄像头)
        data='',  # dataset.yaml路径
        imgsz=(640, 640),  # 推理尺寸 (高度, 宽度)
        conf_thres=0.25,  # 置信度阈值
        iou_thres=0.45,  # NMS IoU阈值
        max_det=1000,  # 每张图片最大检测数
        device='',  # cuda设备，例如 0 或 0,1,2,3 或 cpu
        view_img=False,  # 显示结果
        save_txt=False,  # 保存结果到 *.txt 文件
        save_conf=False,  # 在--save-txt标签中保存置信度
        save_crop=False,  # 保存裁剪的预测框
        save_img=True,
        classes=None,  # 按类别过滤：--class 0, 或 --class 0 2 3
        agnostic_nms=False,  # 类别无关的NMS
        augment=False,  # 增强推理
        visualize=False,  # 可视化特征
        update=False,  # 更新所有模型
        project='result',  # 保存结果到 project/name
        name='',  # 保存结果到 project/name
        exist_ok=True,  # 现有的project/name可以，不增量
        line_thickness=3,  # 边界框厚度（像素）
        hide_labels=False,  # 隐藏标签
        hide_conf=False,  # 隐藏置信度
        half=False,  # 使用FP16半精度推理
        dnn=False,  # 使用OpenCV DNN进行ONNX推理
        vid_stride=1,  # 视频帧率步长
        show_label=None,
        use_camera=False
):
    source = str(source)
    # 目录
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 增量运行
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建目录
    webcam = source.isnumeric()
    # 加载模型
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # 检查图像尺寸

    # 数据加载器
    bs = 1  # 批次大小
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    vid_path, vid_writer = [None] * bs, [None] * bs

    # 运行推理
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # 预热
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 转换为 fp16/32
            im /= 255  # 0 - 255 转换为 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # 为批次维度扩展

        # 推理
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # 非极大值抑制
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # 第二阶段分类器（可选）
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # 处理预测结果
        for i, det in enumerate(pred):  # 每张图片
            seen += 1
            if webcam:  # 批次大小 >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # 转换为Path
            save_path = str(save_dir / p.name)  # im.jpg
            s += '%gx%g ' % im.shape[2:]  # 打印字符串
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # 将框从img_size重新缩放到im0尺寸
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 打印结果
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # 每个类别的检测数
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 添加到字符串

                # 写入结果
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # 整数类别
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # 流式结果
            im0 = annotator.result()
            # 只有当show_label不为None时才更新显示
            if show_label is not None:
                im1 = im0.astype("uint8")
                im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
                
                # 直接使用原始尺寸，让UI组件的setScaledContents自动处理拉伸
                im1 = QtGui.QImage(im1[:], im1.shape[1], im1.shape[0], im1.shape[1] * 3, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap(im1)
                
                # 直接设置pixmap，UI组件会自动拉伸图像撑满显示框
                show_label.setPixmap(pixmap)

            # 保存结果（带检测框的图像）
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # '视频'或'流'
                    if vid_path[i] != save_path:  # 新视频
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # 释放之前的视频写入器
                        if vid_cap:  # 视频
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # 流
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # 强制结果视频使用*.mp4后缀
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # 打印时间（仅推理）
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # 关闭摄像头->退出
        if not Globals.camera_running and use_camera:
            dataset.cap.release()  # 释放摄像头
            break