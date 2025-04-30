import cv2

# 打开视频文件或摄像头
cap = cv2.VideoCapture('video.mp4')  # 如果使用摄像头，请将 'video.mp4' 替换为 0

# 创建背景减除器（以MOG2为例）
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break

    # 应用背景减除器
    fg_mask = back_sub.apply(frame)

    # 使用形态学操作去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # 查找前景物体的轮廓
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # 忽略小面积的噪声
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Frame', frame)
    cv2.imshow('Foreground Mask', fg_mask)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
    # 暂停并保存当前帧和前景掩码
        cv2.imwrite('paused_frame.png', frame)
        cv2.imwrite('paused_fg_mask.png', fg_mask)
        print("Frame and Foreground Mask saved as 'paused_frame.png' and 'paused_fg_mask.png'")
        while True:
            # 等待按下 'p' 键继续
            if cv2.waitKey(30) & 0xFF == ord('p'):
                break

# 释放资源
cap.release()
cv2.destroyAllWindows()
