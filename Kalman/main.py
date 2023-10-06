import cv2, os
import numpy as np
import const
import utils
import measure
from kalman import Kalman
# --------------------------------Kalman參數---------------------------------------
# 狀態轉移矩陣，上一時刻的狀態轉移到當前時刻
A = np.array([[1, 0, 0, 0, 1, 0],
              [0, 1, 0, 0, 0, 1],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])
# 控制輸入矩陣B
B = None
# 過程噪聲協方差矩陣
Q = np.eye(A.shape[0]) * 0.1        #對角線為 0.1
# 狀態觀測矩陣
H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0]])
# 觀測噪聲協方差矩陣
R = np.eye(H.shape[0]) * const.MEASUREMENT_UNCERTAINTY
# 狀態估計協方差矩陣
P = np.eye(A.shape[0])



def main():
    # ========================================================================================
    #                                   影片載入
    # ========================================================================================
    current_directory_path = os.path.dirname(os.path.abspath(__file__))
    cap = cv2.VideoCapture(current_directory_path + const.VIDEO_PATH) 
    meas_list_all = measure.load_measurement(current_directory_path + const.FILE_DIR)
    sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # opencv3.0
    video_writer = cv2.VideoWriter(current_directory_path + const.VIDEO_OUTPUT_PATH, fourcc, const.FPS, sz, True)

    # ========================================================================================
    #                                   逐幀讀取
    # ========================================================================================
    state_list = []  #存kalman 
    frame_cnt = 1
    for meas_list_frame in meas_list_all:
        ret, frame = cap.read()
        if not ret:
            break

        # ========================================================================================
        #                                   Kalman Filter(multi-objects)
        # ========================================================================================
        for target in state_list:
            target.predict()

        mea_list = [utils.box2meas(mea) for mea in meas_list_frame]
        state_rem_list, mea_rem_list, match_list = Kalman.association(state_list, mea_list)     #匹配觀測值
        
        # 狀態值沒沒匹配上，就用預測的方式填補資料
        state_del = list()
        for idx in state_rem_list:
            status, _, _ = state_list[idx].update()
            if not status:
                state_del.append(idx)
        state_list = [state_list[i] for i in range(len(state_list)) if i not in state_del]      #連續好幾筆都沒匹配到就刪除，認定為失去追蹤

        # 觀測值沒匹配到則認定為是新生目標，創建新的 Kalman Filter 給該新生目標使用
        for idx in mea_rem_list:
            state_list.append(Kalman(A, B, H, Q, R, utils.mea2state(mea_list[idx]), P))

        # ========================================================================================
        #                                   畫圖
        # ========================================================================================
        # 顯示(觀測資料)
        for mea in meas_list_frame:
            cv2.rectangle(frame, tuple(mea[:2]), tuple(mea[2:]), const.COLOR_MEA, thickness=1)
        # 顯示(預測資料)
        for kalman in state_list:
            pos = utils.state2box(kalman.X_posterior)
            cv2.rectangle(frame, tuple(pos[:2]), tuple(pos[2:]), const.COLOR_STA, thickness=2)
        # 將匹配關系畫出來(更新幅度、方向)
        for item in match_list:
            cv2.line(frame, tuple(item[0][:2]), tuple(item[1][:2]), const.COLOR_MATCH, 3)
        # 移動軌跡
        for kalman in state_list:
            tracks_list = kalman.track
            for idx in range(len(tracks_list) - 1):
                last_frame = tracks_list[idx]
                cur_frame = tracks_list[idx + 1]
                cv2.line(frame, last_frame, cur_frame, kalman.track_color, 2)

        cv2.putText(frame, str(frame_cnt), (0, 50), color=const.RED, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5)
        cv2.imshow('Demo', frame)
        cv2.imwrite("./image/{}.jpg".format(frame_cnt), frame)
        video_writer.write(frame)
        cv2.waitKey(100)  # 顯示 1000 ms 即 1s 後消失
        frame_cnt += 1

    cap.release()
    cv2.destroyAllWindows()
    video_writer.release()


if __name__ == '__main__':
    main()
