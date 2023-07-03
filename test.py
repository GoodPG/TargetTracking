# coding=utf-8
import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import tkinter.filedialog
import numpy as np
import math
import ctypes

# 设置窗口
user32 = ctypes.windll.user32 
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
window_width = int(screensize[0])
window_height = int(screensize[1]/2-10)
image_width = int(window_width/2-10)
image_height = int(window_height-60)
imagepos_x = 0
imagepos_y = 0
butpos_x = 450
butpos_y = 450
isOpen = 0
history = 100
varThreshold = 40
detectShadows = False
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
background = None
backSub = cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)
g = 0
bg = 0
md = 0
isCanvas = 0

old_gray = None
p0 = None
mask = None

# 设置背景减法器，这里使用opencv提供的MOG2算法
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# 构建结构元素，用于形态学运算
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# ShiTomasi corner detection的参数
feature_params = dict(maxCorners=200,
                      qualityLevel=0.1,
                      minDistance=12,
                      blockSize=7)
# 光流法参数
lk_params = dict(winSize=(30, 30),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
# 创建随机生成的颜色用以绘制跟踪线
color = np.random.randint(0, 255, (200, 3))

# 原视频地址
path_ = ""


# target detection
def targetDetection(frame):
    global bg_subtractor
    global kernel
    # 背景减法
    fg_mask = bg_subtractor.apply(frame)
    # 形态学开运算
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    # 找到所有轮廓
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历所有轮廓，框出目标物体
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    outImgList = [tkImage(frame), tkImage(fg_mask)]
    return outImgList


class EuclideanDistTracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, objects_rect):
        objects_bbs_ids = []
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids
# 创建跟踪器对象
tracker = EuclideanDistTracker()
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

#目标跟踪
def targetTrack(frame):
    global es
    global background
    height, width, _ = frame.shape
    roi = frame

    # 1. 物体检测
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, None, iterations=3)  # 膨胀操作
    mask = cv2.erode(mask, None, iterations=3)  # 腐蚀操作

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        # 计算面积并去除小元素
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    # 2. 对象追踪
    boxes_ids = tracker.update(detections)

    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return tkImage(frame=frame)


# 混合高斯模型背景重建
def GMM(frame):
    global backSub
    if frame is None:
        return None
    fgMask = backSub.apply(frame)
    rows, cols, _channels = map(int, frame.shape)
    fgMasksrc = cv2.pyrDown(fgMask, dstsize=(cols // 2, rows // 2))
    return tkImage(fgMasksrc)

#返回背景图
def GMM_background(frame,frame_gray,background_gray):
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #加权平均法求背景图像
    background_gray = cv2.addWeighted(background_gray,0.98,frame_gray,0.02,0)
    rows, cols, _channels = map(int, frame.shape)
    background_realtime = cv2.pyrDown(background_gray,dstsize=(cols//2,rows//2))
    return tkImage(background_realtime)

# 光流法追踪
def opticalFlow(frame, old_frame):
    global old_gray
    global p0
    global mask

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 计算光流以获取点的新位置
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # 选择good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)
    # cv2.imshow('frame', img)
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    return tkImage(img)


# 打开文件
def openFile():
    # 选择文件,path_接收文件地址
    global path_
    path_ = tkinter.filedialog.askopenfilename()
    print(path_)
    # path设置path_的值
    path.set(path_)
    # video(path=path_)


# 暂停
def pauseView():
    global pause_butt
    global lock
    lock += 1
    if lock % 2 == 0:
        pause_butt.destroy()
        pause_butt = Button(win, text='pause', bd=1, command=pauseView)
        pause_butt.place(x=window_width / 2 + 150, y=10)
    else:
        pause_butt.destroy()
        pause_butt = Button(win, text='continue', bd=1, command=pauseView)
        pause_butt.place(x=window_width / 2 + 150, y=10)


# 背景重建按钮功能
def isGmm():
    global g
    global bg_butt
    if g == 0:
        g = 1
        bg_butt.destroy()
        bg_butt = Button(win, text='关闭背景重建', bd=1, command=isGmm)
        bg_butt.place(x=window_width / 2 + 220, y=10)
    else:
        g = 0
        bg_butt.destroy()
        bg_butt = Button(win, text='打开背景重建', bd=1, command=isGmm)
        bg_butt.place(x=window_width / 2 + 220, y=10)


# 目标检测按钮
def isDetect():
    global bg
    global target_butt
    if bg == 0:
        bg = 1
        target_butt.destroy()
        target_butt = Button(win, text='关闭目标检测', bd=1, command=isDetect)
        target_butt.place(x=window_width / 2 + 320, y=10)
    else:
        bg = 0
        target_butt.destroy()
        target_butt = Button(win, text='打开目标检测', bd=1, command=isDetect)
        target_butt.place(x=window_width / 2 + 320, y=10)


# 遮挡纠错
def isModify():
    global md
    global modify_butt
    if md == 0:
        md = 1
        modify_butt.destroy()
        modify_butt = Button(win, text='关闭遮挡纠错', bd=1, command=isModify)
        modify_butt.place(x=window_width / 2 + 420, y=10)
    else:
        md = 0
        modify_butt.destroy()
        modify_butt = Button(win, text='打开遮挡纠错', bd=1, command=isModify)
        modify_butt.place(x=window_width / 2 + 420, y=10)


def addCanvas():
    global canvas3
    global isCanvas
    global win
    if isCanvas == 0:
        win.geometry(str(window_width) + 'x' + str(window_height + image_height + 10))
        canvas3 = Canvas(win, bg='gray', width=image_width, height=image_height)
        canvas3.place(x=window_width - image_width * 3 / 2, y=60 + image_height)
        isCanvas = 1
    else:
        return


def updateCanvas(picture):
    global canvas3
    canvas3.create_image(0, 0, anchor='nw', image=picture)


def destroyCanvas():
    global canvas3
    global isCanvas
    global win
    if isCanvas == 1:
        win.geometry(str(window_width) + 'x' + str(window_height))
        canvas3.destroy()
        isCanvas = 0
    else:
        return


# start,请看这里，请看这里,请看这里，请看这里,请看这里，请看这里,请看这里，请看这里,请看这里，请看这里
def startView():
    global old_gray
    global p0
    global mask

    vc1 = cv2.VideoCapture(path_)  # 打开 path_ 路径的视频

    old_frame = tkFrame(vc=vc1)  # 取出视频的第一帧
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  # 灰度化
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    mask = np.zeros_like(old_frame)  # 为绘制创建掩码图片
    ret,background_color = vc1.read()
    bk_gray = cv2.cvtColor(background_color, cv2.COLOR_BGR2GRAY)

    try:
        while True:
            if lock % 2 == 0:
                frame1 = tkFrame(vc=vc1)
                # 原视频从视频中抓取帧并转换为图片，我用的ImageTk.PhotoImage 对象
                if bg == 1:
                    picture1 = targetDetection(frame=frame1)[0]
                else:
                    picture1 = tkImage(frame=frame1)
                    # 处理后结果，把处理后的帧传到这里即可,我现在是使用的原视频
                # 背景重建
                if g == 1:

                    picture2 = GMM(frame=frame1)

                    frame_gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
                    bk_gray = cv2.addWeighted(bk_gray, 0.98,frame_gray, 0.02, 0)
                    picture3 = GMM_background(frame=frame1,frame_gray=frame_gray,background_gray=bk_gray)
                    addCanvas()
                    updateCanvas(picture=picture2)
                    updateCanvas(picture=picture3)
                # 目标识别
                elif bg == 1:
                    picture2 = targetDetection(frame=frame1)[1]
                    destroyCanvas()
                # 遮挡纠错
                elif md == 1:
                    picture2 = opticalFlow(frame=frame1, old_frame=old_frame)
                    destroyCanvas()
                else:
                    picture2 = targetTrack(frame=frame1)
                    destroyCanvas()
                canvas1.create_image(0, 0, anchor='nw', image=picture1)  # 原视频
                canvas2.create_image(0, 0, anchor='nw', image=picture2)  # 处理后视频
                win.update_idletasks()  # 最重要的更新是靠这两句来实现
                win.update()
            else:
                win.update_idletasks()  # 最重要的更新是靠这两句来实现
                win.update()
    except:
        pass
    vc1.release()
    cv2.destroyAllWindows()


# 抓取视频的帧
def tkFrame(vc):
    ref, frame = vc.read()
    return frame


# 图像转换，用于在画布中显示
def tkImage(frame):
    cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(cvimage)
    pilImage = pilImage.resize((image_width, image_height), Image.BILINEAR)
    tkImage = ImageTk.PhotoImage(image=pilImage)
    return tkImage


# 图像的显示与更新
# def video(picture1=None,picture2=None):
#     if lock % 2 == 0:
#         canvas1.create_image(0,0,anchor='nw',image=picture1)  #原视频
#         canvas2.create_image(0,0,anchor='nw',image=picture2)   #处理后视频
#         win.update_idletasks()  #最重要的更新是靠这两句来实现
#         win.update()
#     else:
#         win.update_idletasks()  #最重要的更新是靠这两句来实现
#         win.update()
#    win.mainloop()
#    vc1.release()
#    cv2.destroyAllWindows()

'''UI布局'''
lock = 0  # 暂停标志
# 绘制窗口以及设置窗口几何
win = tk.Tk()
win.geometry(str(window_width) + 'x' + str(window_height)+'+0+0')
win.title("运动目标检测与跟踪")
top = Menu(win)
top.add_command(label="打开", command=openFile)
win.config(menu=top)
# 设置功能按钮
open_butt = Button(win, text="open", bd=1, command=openFile)
open_butt.place(x=window_width/10, y=10)
start_butt = Button(win, text='start', bd=1, command=startView)
start_butt.place(x=window_width / 2 + 100, y=10)
pause_butt = Button(win, text='pause', bd=1, command=pauseView)
pause_butt.place(x=window_width / 2 + 150, y=10)
bg_butt = Button(win, text='打开背景重建', bd=1, command=isGmm)
bg_butt.place(x=window_width / 2 + 220, y=10)
target_butt = Button(win, text='打开目标检测', bd=1, command=isDetect)
target_butt.place(x=window_width / 2 + 320, y=10)
modify_butt = Button(win, text='打开遮挡纠错', bd=1, command=isModify)
modify_butt.place(x=window_width / 2 + 420, y=10)
# 显示文件地址
path = tk.StringVar()
dir_file = Entry(win, text=path, width=70)
dir_file.place(x=window_width/10+50, y=10)

canvas1 = Canvas(win, bg='gray', width=image_width, height=image_height)
canvas1.place(x=imagepos_x, y=50)
canvas2 = Canvas(win, bg='gray', width=image_width, height=image_height)
canvas2.place(x=window_width - image_width - 2, y=50)
canvas3 = None

win.mainloop()
startView()
# if __name__ == '__main__':
#     p1 = multiprocessing.Process(target=video)
#     p1.start()