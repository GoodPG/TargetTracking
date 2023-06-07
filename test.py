import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import tkinter.filedialog
import numpy as np
#设置窗口
window_width=2048
window_height=640
image_width=int(1024)
image_height=int(576)
imagepos_x=0
imagepos_y=0
butpos_x=450
butpos_y=450
isOpen =0
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
background = None
backSub = cv2.createBackgroundSubtractorMOG2()
g=0
bg=0
md=0
# 设置背景减法器，这里使用opencv提供的MOG2算法
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# 构建结构元素，用于形态学运算
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

#原视频地址
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
    outImgList = [tkImage(frame),tkImage(fg_mask)]
    return outImgList

#目标跟踪
def targetTrack(frame_lwpCV):
    global es
    global background
    gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
    gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)

    # 将第一帧设置为整个输入的背景
    if background is None:
        background = gray_lwpCV
    # 对于每个从背景之后读取的帧都会计算其与北京之间的差异，并得到一个差分图（different map）。
    # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
    diff = cv2.absdiff(background, gray_lwpCV)
    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]  # 二值化阈值处理
    diff = cv2.dilate(diff, es, iterations=2)  # 形态学膨胀

    # 显示矩形框
    contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)  # 该函数计算一幅图像中目标的轮廓
    for c in contours:
        if cv2.contourArea(c) < 150:  # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
            continue
        (x, y, w, h) = cv2.boundingRect(c)  # 该函数计算矩形的边界框
        cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return tkImage(frame=frame_lwpCV)


#混合高斯模型背景重建
def GMM(frame):
    global backSub
    if frame is None:
        return None
    fgMask = backSub.apply(frame)
    rows,cols,_channels = map(int,frame.shape)
    fgMasksrc = cv2.pyrDown(fgMask,dstsize=(cols//2,rows//2))
    return tkImage(fgMasksrc) 


#打开文件
def openFile():
    #选择文件,path_接收文件地址
    global path_
    path_ = tkinter.filedialog.askopenfilename()
    print(path_)
    #path设置path_的值
    path.set(path_)
    #video(path=path_)

#暂停
def pauseView():
    global pause_butt
    global lock
    lock+=1
    if lock % 2 == 0:
        pause_butt.destroy()
        pause_butt = Button(win,text='pause',bd=1,command=pauseView)
        pause_butt.place(x=window_width/2+30,y=10)
    else:
        pause_butt.destroy()
        pause_butt = Button(win,text='continue',bd=1,command=pauseView)
        pause_butt.place(x=window_width/2+30,y=10)


#背景重建按钮功能
def isGmm():
    global g
    global bg_butt
    if g == 0:
        g=1
        bg_butt.destroy()
        bg_butt = Button(win,text='关闭背景重建',bd=1,command=isGmm)
        bg_butt.place(x=window_width/2+120,y=10)
    else:
        g=0
        bg_butt.destroy()
        bg_butt = Button(win,text='打开背景重建',bd=1,command=isGmm)
        bg_butt.place(x=window_width/2+120,y=10)


#目标检测按钮
def isDetect():
    global bg
    global target_butt
    if bg == 0:
        bg = 1
        target_butt.destroy()
        target_butt = Button(win,text='关闭目标检测',bd=1,command=isDetect)
        target_butt.place(x=window_width/2+250,y=10)
    else:
        bg = 0
        target_butt.destroy()
        target_butt = Button(win,text='打开目标检测',bd=1,command=isDetect)
        target_butt.place(x=window_width/2+250,y=10)


#遮挡纠错
def isModify():
    global md
    global modify_butt
    if md == 0:
        md = 1
        modify_butt.destroy()
        modify_butt = Button(win,text='关闭遮挡纠错',bd=1,command=isModify)
        modify_butt.place(x=window_width/2+400,y=10)
    else:
        md = 0
        modify_butt.destroy()
        modify_butt = Button(win,text='打开遮挡纠错',bd=1,command=isModify)
        modify_butt.place(x=window_width/2+400,y=10)

#start,请看这里，请看这里,请看这里，请看这里,请看这里，请看这里,请看这里，请看这里,请看这里，请看这里
def startView():
    vc1 = cv2.VideoCapture(path_)  #打开 path_ 路径的视频
    try:
        while True:
            if lock % 2 == 0:
                frame1 = tkFrame(vc=vc1)
                #原视频从视频中抓取帧并转换为图片，我用的ImageTk.PhotoImage 对象
                if bg == 1:
                    picture1=targetDetection(frame=frame1)[0]
                else:
                    picture1=tkImage(frame=frame1)  
                #处理后结果，把处理后的帧传到这里即可,我现在是使用的原视频
                if g == 1:
                    picture2=GMM(frame=frame1)
                elif bg == 1:
                    picture2=targetDetection(frame=frame1)[1]
                elif md == 1:
                    picture2=None
                else:
                    picture2=targetTrack(frame_lwpCV=frame1)                     
                canvas1.create_image(0,0,anchor='nw',image=picture1)  #原视频
                canvas2.create_image(0,0,anchor='nw',image=picture2)   #处理后视频
                win.update_idletasks()  #最重要的更新是靠这两句来实现
                win.update()
            else:
                win.update_idletasks()  #最重要的更新是靠这两句来实现
                win.update()
    except:
        pass
    vc1.release()
    cv2.destroyAllWindows()

#抓取视频的帧
def tkFrame(vc):
    ref,frame = vc.read()
    return frame

#图像转换，用于在画布中显示
def tkImage(frame):
    cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(cvimage)
    pilImage = pilImage.resize((image_width, image_height),Image.ANTIALIAS)
    tkImage =  ImageTk.PhotoImage(image=pilImage)
    return tkImage

#图像的显示与更新
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
lock=0  #暂停标志
#绘制窗口以及设置窗口几何
win = tk.Tk()
win.geometry(str(window_width)+'x'+str(window_height))
#设置功能按钮
open_butt = Button(win,text="open",bd=1,command=openFile)
open_butt.place(x=360,y=10)
start_butt = Button(win,text='start',bd=1,command=startView)
start_butt.place(x=window_width/2-20,y=10)
pause_butt = Button(win,text='pause',bd=1,command=pauseView)
pause_butt.place(x=window_width/2+30,y=10)
bg_butt = Button(win,text='打开背景重建',bd=1,command=isGmm)
bg_butt.place(x=window_width/2+120,y=10)
target_butt = Button(win,text='打开目标检测',bd=1,command=isDetect)
target_butt.place(x=window_width/2+250,y=10)
modify_butt = Button(win,text='打开遮挡纠错',bd=1,command=isModify)
modify_butt.place(x=window_width/2+400,y=10)
#显示文件地址
path=tk.StringVar()
dir_file = Entry(win,text=path,width=70)
dir_file.place(x=420,y=10)

canvas1 =Canvas(win,bg='gray',width=image_width,height=image_height)
canvas1.place(x=imagepos_x,y=50)
canvas2 =Canvas(win,bg='gray',width=image_width,height=image_height)
canvas2.place(x=1024,y=50)

win.mainloop()
startView()
# if __name__ == '__main__':
#     p1 = multiprocessing.Process(target=video)
#     p1.start()