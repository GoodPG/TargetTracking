import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import tkinter.filedialog

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

#原视频地址
path_ = ""

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

#start,请看这里，请看这里,请看这里，请看这里,请看这里，请看这里,请看这里，请看这里,请看这里，请看这里
def startView():
    vc1 = cv2.VideoCapture(path_)  #打开 path_ 路径的视频
    try:
        while True:
            if lock % 2 == 0:
                picture1=tkImage(vc1)  #原视频从视频中抓取帧并转换为图片，我用的ImageTk.PhotoImage 对象
                picture2=picture1      #处理后结果，把处理后的帧传到这里即可,我现在是使用的原视频
                canvas1.create_image(0,0,anchor='nw',image=picture1)  #原视频
                canvas2.create_image(0,0,anchor='nw',image=picture2)   #处理后视频
                win.update_idletasks()  #最重要的更新是靠这两句来实现
                win.update()
            else:
                win.update_idletasks()  #最重要的更新是靠这两句来实现
                win.update()
    except:
        pass
    cv2.destroyAllWindows()

#图像转换，用于在画布中显示
def tkImage(vc):
    ref,frame = vc.read()
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
bg_butt = Button(win,text='打开背景重建',bd=1,command=None)
bg_butt.place(x=window_width/2+120,y=10)
target_butt = Button(win,text='关闭目标检测与跟踪',bd=1,command=None)
target_butt.place(x=window_width/2+250,y=10)
modify_butt = Button(win,text='关闭遮挡纠错',bd=1,command=None)
modify_butt.place(x=window_width/2+450,y=10)
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