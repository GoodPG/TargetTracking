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
    #选择文件path_接收文件地址
    path_ = tkinter.filedialog.askopenfilename()
    
    #注意：\\转义后为\
    path_=path_.replace("/","\\\\")
    print(path_)
    #path设置path_的值
    path.set(path_)
    video(path=path_)

#暂停
def pauseView():
    global lock
    lock+=1

#图像转换，用于在画布中显示
def tkImage(vc):
    ref,frame = vc.read()
    cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(cvimage)
    pilImage = pilImage.resize((image_width, image_height),Image.ANTIALIAS)
    tkImage =  ImageTk.PhotoImage(image=pilImage)
    return tkImage

#图像的显示与更新
def video(path,picture=None):
    def video_loop():
       vc1 = cv2.VideoCapture(path)
       try:
            while True:
                if lock % 2 == 0:
                    picture1=tkImage(vc1)
                    canvas1.create_image(0,0,anchor='nw',image=picture1)  #原视频
                    canvas2.create_image(0,0,anchor='nw',image=picture)   #处理后视频
                    win.update_idletasks()  #最重要的更新是靠这两句来实现
                    win.update()
                else:
                    win.update_idletasks()  #最重要的更新是靠这两句来实现
                    win.update()
       except:
            pass
          
    video_loop()
    win.mainloop()
    vc1.release()
    cv2.destroyAllWindows()

'''UI布局'''
lock=0  #暂停标志
#绘制窗口以及设置窗口几何
win = tk.Tk()
win.geometry(str(window_width)+'x'+str(window_height))
#设置功能按钮
open_butt = Button(win,text="open",bd=1,command=openFile)
open_butt.place(x=360,y=10)
pause_butt = Button(win,text='pause',bd=1,command=pauseView)
pause_butt.place(x=window_width/2-20,y=10)
#显示文件地址
path=tk.StringVar()
dir_file = Entry(win,text=path,width=70)
dir_file.place(x=420,y=10)

canvas1 =Canvas(win,bg='white',width=image_width,height=image_height)
canvas1.place(x=imagepos_x,y=50)
canvas2 =Canvas(win,bg='white',width=image_width,height=image_height)
canvas2.place(x=1024,y=50)

video('')
# if __name__ == '__main__': 
#     p1 = multiprocessing.Process(target=video)
#     p1.start()