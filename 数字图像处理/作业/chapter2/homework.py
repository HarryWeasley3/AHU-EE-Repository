import numpy as np
from PIL import Image
import math

def rotate_image(image, angle_deg, interpolation='bilinear'):
    """
    图像任意角度旋转函数
    :param image: PIL Image 输入图像
    :param angle_deg: 旋转角度（顺时针方向）
    :param interpolation: 插值方法 ('nearest', 'bilinear', 'bicubic')
    :return: 旋转后的 PIL Image
    """
    # 将角度转换为弧度（并处理顺时针方向）
    angle_rad = -math.radians(angle_deg)
    
    # 原始图像参数
    src_img = np.array(image)
    h, w, c = src_img.shape
    center_x, center_y = w/2, h/2  # 旋转中心设为图像中心

    # 计算旋转后的画布大小
    cos_theta = abs(math.cos(angle_rad))
    sin_theta = abs(math.sin(angle_rad))
    new_w = int(w * cos_theta + h * sin_theta)
    new_h = int(w * sin_theta + h * cos_theta)
    new_center_x, new_center_y = new_w/2, new_h/2

    # 创建目标图像
    dst_img = np.zeros((new_h, new_w, c), dtype=np.uint8)

    # 反向映射：遍历目标图像的每个像素
    for y in range(new_h):
        for x in range(new_w):
            # 将坐标平移到中心
            x_centered = x - new_center_x
            y_centered = y - new_center_y

            # 逆向旋转（反向映射）
            x_rot = x_centered * math.cos(angle_rad) - y_centered * math.sin(angle_rad)
            y_rot = x_centered * math.sin(angle_rad) + y_centered * math.cos(angle_rad)

            # 平移回原图坐标系
            src_x = x_rot + center_x
            src_y = y_rot + center_y

            # 根据插值方法获取像素值
            if interpolation == 'nearest':
                dst_img[y, x] = nearest_neighbor(src_img, src_x, src_y)
            elif interpolation == 'bilinear':
                dst_img[y, x] = bilinear_interpolation(src_img, src_x, src_y)
            elif interpolation == 'bicubic':
                dst_img[y, x] = bicubic_interpolation(src_img, src_x, src_y)

    return Image.fromarray(dst_img)

#--------------------------------------------------
# 三种插值方法的实现
#--------------------------------------------------
def nearest_neighbor(img, x, y):
    """ 最近邻插值 """
    x_round = int(round(x))
    y_round = int(round(y))
    
    # 边界检查
    if x_round < 0 or x_round >= img.shape[1] or y_round < 0 or y_round >= img.shape[0]:
        return np.zeros(img.shape[2], dtype=np.uint8)  # 返回黑色像素
    
    return img[y_round, x_round]

def bilinear_interpolation(img, x, y):
    """ 双线性插值 """
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1
    
    # 边界处理
    x0 = np.clip(x0, 0, img.shape[1]-1)
    x1 = np.clip(x1, 0, img.shape[1]-1)
    y0 = np.clip(y0, 0, img.shape[0]-1)
    y1 = np.clip(y1, 0, img.shape[0]-1)
    
    # 计算权重
    dx = x - x0
    dy = y - y0
    w00 = (1 - dx) * (1 - dy)
    w01 = (1 - dx) * dy
    w10 = dx * (1 - dy)
    w11 = dx * dy
    
    # 加权平均
    return np.clip(
        w00 * img[y0, x0] + w01 * img[y1, x0] + 
        w10 * img[y0, x1] + w11 * img[y1, x1], 
        0, 255
    ).astype(np.uint8)

def cubic_convolution(a, x):
    """ 三次卷积核函数 (a=-0.75时效果较好) """
    x_abs = abs(x)
    if x_abs <= 1:
        return (a + 2)*x_abs**3 - (a + 3)*x_abs**2 + 1
    elif x_abs < 2:
        return a*x_abs**3 - 5*a*x_abs**2 + 8*a*x_abs - 4*a
    else:
        return 0

def bicubic_interpolation(img, x, y):
    """ 双三次插值 """
    x_floor = int(math.floor(x))
    y_floor = int(math.floor(y))
    
    # 获取16邻域像素
    pixels = []
    for i in range(-1, 3):
        for j in range(-1, 3):
            px = x_floor + j
            py = y_floor + i
            # 边界处理（镜像）
            px = np.clip(px, 0, img.shape[1]-1)
            py = np.clip(py, 0, img.shape[0]-1)
            pixels.append(img[py, px])
    
    # 计算水平/垂直权重
    dx = x - x_floor
    dy = y - y_floor
    
    wx = [cubic_convolution(-0.75, dx + 1),
          cubic_convolution(-0.75, dx),
          cubic_convolution(-0.75, 1 - dx),
          cubic_convolution(-0.75, 2 - dx)]
    
    wy = [cubic_convolution(-0.75, dy + 1),
          cubic_convolution(-0.75, dy),
          cubic_convolution(-0.75, 1 - dy),
          cubic_convolution(-0.75, 2 - dy)]
    
    # 计算加权值
    result = np.zeros(img.shape[2], dtype=np.float32)
    for i in range(4):
        for j in range(4):
            weight = wx[j] * wy[i]
            result += weight * pixels[i*4 + j]
    
    return np.clip(result, 0, 255).astype(np.uint8)

#--------------------------------------------------
# 使用示例
#--------------------------------------------------
if __name__ == "__main__":
    # 加载测试图像
    img = Image.open("test.jpg")
    
    # 旋转45度，使用不同插值方法
    img_nearest = rotate_image(img, 45, 'nearest')
    img_nearest.save("rotated_nearest.jpg")
    img_bilinear = rotate_image(img, 45, 'bilinear')
    img_bilinear.save("rotated_bilinear.jpg")
    img_bicubic = rotate_image(img, 45, 'bicubic')
    
    # 保存结果
    # img_nearest.save("rotated_nearest.jpg")
    # img_bilinear.save("rotated_bilinear.jpg")
    img_bicubic.save("rotated_bicubic.jpg")