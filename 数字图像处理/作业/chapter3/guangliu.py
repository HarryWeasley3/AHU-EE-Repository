import cv2
import numpy as np

# 打开视频文件或摄像头
cap = cv2.VideoCapture('video.mp4')  # 如果使用摄像头，请将'video.mp4'替换为0

# 读取第一帧并将其转换为灰度图像
ret, old_frame = cap.read()
if not ret:
    print("无法读取视频源")
    cap.release()
    cv2.destroyAllWindows()
    exit()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# 使用Shi-Tomasi角点检测找到初始特征点
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# 设置Lucas-Kanade光流参数
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 创建用于绘制轨迹的颜色
color = np.random.randint(0, 255, (100, 3))

# 创建掩膜用于绘制轨迹
mask = np.zeros_like(old_frame)

while True:
    # 读取新帧并将其转换为灰度图像
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 计算光流
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # 选择好的跟踪点
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

    # 绘制跟踪结果
   # 绘制跟踪结果
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

    img = cv2.add(frame, mask)

    # 显示结果
    cv2.imshow('Optical Flow', img)

    # 按下'q'键退出循环
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # 更新前一帧和特征点
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# 释放资源
cap.release()
cv2.destroyAllWindows()
