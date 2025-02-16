from ultralytics import YOLO
import pygetwindow
import cv2
import numpy as np
import torch
from PIL import ImageGrab
#加载模型
model = YOLO(r"C:\Users\zjq\Desktop\yolov10\yolov10-main\runs\detect\train\weights\best.pt")#刚刚训练好的
 
# #获取窗口 旧版本的pygetwindow库用
# window_name = "Apex Legends"
# window = pygetwindow.getWindowTitle(window_name)[0]
# 获取所有窗口
all_windows = pygetwindow.getAllWindows()
 
# 假设你知道窗口标题，可以通过标题筛选窗口
target_window_title = "Apex Legends"
for window in all_windows:
    if window.title == target_window_title:
        print(f"Found window: {window.title}")
        # 可以对找到的窗口进行操作，比如激活
        window.activate()
        # window = window[0]
        break
else:
    print(f"Window with title '{target_window_title}' not found.")
###
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame", 1280, 800)
while True:
    if window :
        x, y ,w, h = window.left, window.top, window.width, window.height
        screenshot = ImageGrab.grab(bbox=(x, y, x+w, y+h))
        image_src=cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2RGB)
        size_x, size_y = image_src.shape[1], image_src.shape[0]
        image_det=cv2.resize(image_src, (1920, 1080))
        results = model.predict(source=image_det, imgsz=640, conf=0.7,
                                save=False)
 
 
        boxes=results[0].boxes.xywhn
        for box in boxes:
            #坐标转换
            x_coordinate_top_left=int((box[0]-box[2]/2)*size_x)
            y_coordinate_top_left=int((box[1]-box[3]/2)*size_x)
            x_coordinate_bottom_right=int((box[0]+box[2]/2)*size_x)
            y_coordinate_bottom_right=int((box[1]+box[3]/2)*size_y)
            #cv2.rectangle(image_src,(x_coordinate_top_left,y_coordinate_top_left),(x_coordinate_bottom_right,y_coordinate_bottom_right),color=(0,255,0),thickness=3)
            cv2.rectangle(image_src,(x_coordinate_top_left,y_coordinate_top_left-200),(x_coordinate_bottom_right,y_coordinate_bottom_right),color=(0,255,0),thickness=3)
 
            cv2.imshow("frame", image_src)
            cv2.waitKey(1)
        cv2.imshow("frame", image_src)
        cv2.waitKey(1)
    else:
        break