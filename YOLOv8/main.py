import cv2
import mss
import torch
from ultralytics import YOLO
import numpy as np
import time
import os


def main():

    current_directory_path = os.path.dirname(os.path.abspath(__file__))
    parent_directory_path = os.path.dirname(current_directory_path)
    cap = cv2.VideoCapture(current_directory_path + r"\Data\testvideo.mp4")         #testvideo    test_block

    frame_cnt = 1

    model = YOLO("yolov8l.pt")

    last_time = time.time()

    while True:
        # 讀取影片
        ret, frame = cap.read()
        if not ret:
            break

        # model 標記
        for result in model.track(source=frame, show=True, stream=True, agnostic_nms=True,device=0):
            #印出資料，看看有甚麼可以用
            # print(result.boxes)

            ClassResult = result.boxes.cls
            indices = torch.nonzero(ClassResult == 0)
            indices = indices.squeeze()

            # print(f"Ori_Cls : {ClassResult}")
            # print(f"Aft_Cls : {indices}")

            # print(f"Ori_xywh : {result.boxes.xywh}")
            # print(f"Aft_xywh : {result.boxes.xywh[indices]}")
            ResultLabel = result.boxes.xyxy[indices]                        #換成 "左" "上" "右" "下"

            ORIframe = result.orig_img


            output_path = parent_directory_path + "/Kalman/Data/Labels/Labels_"+ str(frame_cnt) +".txt"
            with open(output_path, 'w') as file:
            # with open(current_directory_path + '/Labels/testvideo1_'+ str(frame_cnt) +'.txt', 'w') as file:
                for i in range(ResultLabel.size(0)):
                    line = f"{0} {' '.join(str(round(x.item())) for x in ResultLabel[i])}"
                    print(line)
                    file.write(line + '\n')
                    
            frame_cnt = frame_cnt + 1


        # cv2.waitKey(0)
        print("fps: {}".format(1 / (time.time() - last_time)))


if __name__ == "__main__":
    main()

