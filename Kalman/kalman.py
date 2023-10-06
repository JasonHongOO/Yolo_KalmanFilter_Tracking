import random
import numpy as np
import utils
from matcher import Matcher
import const

GENERATE_SET = 1 
TERMINATE_SET = const.TERMINATE_SET


class Kalman:
    def __init__(self, A, B, H, Q, R, X, P):
        # 固定參數
        self.A = A  # 狀態轉移矩陣
        self.B = B  # 控制矩陣
        self.H = H  # 觀測矩陣
        self.Q = Q  # "過程" 噪聲
        self.R = R  # "觀測' 噪聲
        # 叠代參數
        self.X_posterior = X  # 後驗狀態, 定義為 [中心x,中心y,寬w,高h,dx,dy]
        self.P_posterior = P  # 後驗誤差矩陣
        self.X_prior = None  # 先驗狀態
        self.P_prior = None  # 先驗誤差矩陣
        self.K = None  # kalman gain
        self.Z = None  # 觀測資料, 定義為 [中心x,中心y,寬w,高h] 
        self.terminate_count = TERMINATE_SET
        # (暫存)移動軌跡
        self.track = []
        self.track_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.__record_track()

    def predict(self):
        self.X_prior = np.dot(self.A, self.X_posterior)                                                             #少了 Bu !!
        self.P_prior = np.dot(np.dot(self.A, self.P_posterior), self.A.T) + self.Q
        return self.X_prior, self.P_prior

    @staticmethod
    def association(kalman_list, mea_list):
       
        state_rec = {i for i in range(len(kalman_list))}
        mea_rec = {i for i in range(len(mea_list))}

        state_list = list()
        for kalman in kalman_list:
            state = kalman.X_prior
            state_list.append(state[0:4])

        match_dict = Matcher.match(state_list, mea_list)

        state_used = set()
        mea_used = set()
        match_list = list() 
        for state, mea in match_dict.items():
            state_index = int(state.split('_')[1])
            mea_index = int(mea.split('_')[1])
            match_list.append([utils.state2box(state_list[state_index]), utils.mea2box(mea_list[mea_index])])
            kalman_list[state_index].update(mea_list[mea_index])
            state_used.add(state_index)
            mea_used.add(mea_index)

        return list(state_rec - state_used), list(mea_rec - mea_used), match_list

    def update(self, mea=None):

        status = True
        if mea is not None:
            self.Z = mea
            self.K = np.dot(np.dot(self.P_prior, self.H.T),
                            np.linalg.inv(np.dot(np.dot(self.H, self.P_prior), self.H.T) + self.R))  
            self.X_posterior = self.X_prior + np.dot(self.K, self.Z - np.dot(self.H, self.X_prior))  
            self.P_posterior = np.dot(np.eye(6) - np.dot(self.K, self.H), self.P_prior) 
            status = True
            self.terminate_count = TERMINATE_SET
        else:
            if self.terminate_count == 1:
                status = False                          # state 為 false 表示該刪除物體了
            else:
                self.terminate_count -= 1               # 7 次沒有收到 "觀測" 資料就判斷該物體已經不見
                # print(f"terminate_count : {self.terminate_count}")
                self.X_posterior = self.X_prior
                self.P_posterior = self.P_prior
                status = True
        if status:
            self.__record_track()

        return status, self.X_posterior, self.P_posterior

    def __record_track(self):
        self.track.append([int(self.X_posterior[0]), int(self.X_posterior[1])])


if __name__ == '__main__':
    A = np.array([[1, 0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0, 1],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])
    B = None
    Q = np.eye(A.shape[0]) * 0.1
    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0]])
    R = np.eye(H.shape[0]) * 1
    P = np.eye(A.shape[0])

    box = [729, 238, 764, 339]
    X = utils.box2state(box)

    k1 = Kalman(A, B, H, Q, R, X, P)
    print(k1.predict())

    mea = [730, 240, 766, 340]
    mea = utils.box2meas(mea)
    print(k1.update(mea))
