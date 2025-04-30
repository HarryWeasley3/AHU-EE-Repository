import cv2

# 打开视频文件或摄像头
cap = cv2.VideoCapture('video.mp4')  # 如果使用摄像头，请将'video.mp4'替换为0

# 读取第一帧
ret, frame1 = cap.read()
if not ret:
    print("无法读取视频源")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# 将第一帧转换为灰度图像
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:
    # 读取下一帧
    ret, frame2 = cap.read()
    if not ret:
        break

    # 将当前帧转换为灰度图像
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 计算两帧之间的差分
    diff = cv2.absdiff(gray1, gray2)

    # 对差分图像进行阈值处理
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # 使用形态学操作去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # 忽略小面积的噪声
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Frame', frame2)
    cv2.imshow('Thresh', thresh)


    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        # 暂停并保存当前帧和前景掩码
        cv2.imwrite('frame2.png', frame2)
        cv2.imwrite('thresh.png', thresh)
        print("Frame and Foreground Mask saved as 'frame2.png' and 'thresh.png'")
        while True:
            # 等待按下 'p' 键继续
            if cv2.waitKey(30) & 0xFF == ord('p'):
                break
    # 更新前一帧
    gray1 = gray2

# 释放资源
cap.release()
cv2.destroyAllWindows()
