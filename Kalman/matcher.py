import networkx as nx
import numpy as np
import utils
 

class Matcher:
    def __init__(self):
        pass

    @classmethod
    def match(cls, state_list, measure_list):  
        graph = nx.Graph()

        #建一張 Bipartite Graph(二分圖)，每一個 "已存在的object" 去連接所有的 "觀測資料"
        for idx_sta, state in enumerate(state_list):
            state_node = 'state_%d' % idx_sta
            graph.add_node(state_node, bipartite=0)
            for idx_mea, measure in enumerate(measure_list):
                mea_node = 'mea_%d' % idx_mea
                graph.add_node(mea_node, bipartite=1)
                score = cls.cal_iou(state, measure)
                if score is not None:
                    graph.add_edge(state_node, mea_node, weight=score)

        #取出 weight 最大的配對
        match_set = nx.max_weight_matching(graph)  
        res = dict()    #儲存成字典的型態
        for (node_1, node_2) in match_set:
            if node_1.split('_')[0] == 'mea':
                node_1, node_2 = node_2, node_1
            res[node_1] = node_2
        return res

    @classmethod
    def cal_iou(cls, state, measure):
        #資料在有重疊時 inter_h、inter_W 才不會為 0

        state = utils.mea2box(state)  #預測值轉換成 "方框座標"
        measure = utils.mea2box(measure)    #觀測值轉換成 "方框座標"
        s_tl_x, s_tl_y, s_br_x, s_br_y = state[0], state[1], state[2], state[3]     #左上, 右下             #top-left x-coordinate、top-left y-coordinate、bottom-right x-coordinate、bottom-right y-coordinate
        m_tl_x, m_tl_y, m_br_x, m_br_y = measure[0], measure[1], measure[2], measure[3]
        x_min = max(s_tl_x, m_tl_x)
        x_max = min(s_br_x, m_br_x)
        y_min = max(s_tl_y, m_tl_y)
        y_max = min(s_br_y, m_br_y)
        inter_h = max(y_max - y_min + 1, 0) # +1 補救那些非常靠近到沒有重疊
        inter_w = max(x_max - x_min + 1, 0)
        inter = inter_h * inter_w       #重疊部分的面積，重疊面積越大，分數就越高
        if inter == 0:
            return 0
        else:
            return inter / ((s_br_x - s_tl_x) * (s_br_y - s_tl_y) + (m_br_x - m_tl_x) * (m_br_y - m_tl_y) - inter)  #重疊面積 / 未重疊面積


if __name__ == '__main__':
    state_list = []
    measure_list = [np.array([12, 12, 5, 5]).T, np.array([28, 28, 5, 5]).T]
    match_dict = Matcher.match(state_list, measure_list)
    print(match_dict)
    for state, mea in match_dict.items():
        print(state, mea)

