# Yolo_KalmanFilter_Tracking
使用 Yolov8 Model 對影片每一幀的所有人物做標記，接著將資料給 Kalman Filter
結合二分圖去做觀測目標的匹配，實現多目標追蹤的效果。

# Demo 影片 
[![IMAGE ALT TEXT](http://img.youtube.com/vi/eUEREo6-z5A/0.jpg)](https://www.youtube.com/watch?v=eUEREo6-z5A "KalmanFilter Demo")

# 備註
影片預設畫質為 (720 * 360)，可以在 const 中自訂影片輸出大小
不過原始影片就只有 (768 * 576)，設再大也沒幫助
