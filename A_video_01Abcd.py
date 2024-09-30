import cv2
import os
import numpy as np
from PIL import Image
from moviepy.editor import ImageSequenceClip
import re
import random

def vedio_handle1(root_path, vedio_name):
    # 把每一帧提取出来放入文件夹
    vedio_picture(root_path, vedio_name)

    # 图片处理成黑白
    picture_gray(root_path)

    # 对灰度图提取边缘
    picture_gray_edge(root_path)

    # 对边缘图像进行膨胀操作
    edge_pengzhang(root_path)


def vedio_handle2(root_path, vedio_name, create_picture_size, cut_1, cut_2, scale):
    # 生成一张底片
    create_picture(create_picture_size)

    # 切割图片到最佳尺寸
    picture_cut(root_path, cut_1, cut_2)

    # 开始贴图片成为01
    picture_edge_to_01(root_path, create_picture_size, scale)

    # 把图片连接成视频
    picture_to_veido(root_path, vedio_name)


def show_vedio(root_path, vedio_name):
    final_vedio_show(root_path, vedio_name)


def find_up_down_edge(root_path, vedio_up_long, vedio_find_edge_short):
    picture_path = root_path + "picture_0"
    files = os.listdir(picture_path)  # 读入文件夹
    num_png = len(files)  # 统计文件夹中的文件个数

    edge_path = root_path + "picture_2_edge/"
    max_num = 0
    min_num = vedio_up_long
    for i in range(num_png):
        print(i)
        image_path = os.path.join(edge_path, f'{i}.jpg')
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 确保以灰度模式读取
        if image is None:
            print("无法读取图片")
        else:
            for j in range(vedio_up_long):
                if image[j][vedio_find_edge_short] != 0:
                    if max_num < j:
                        max_num = j
                    if min_num > j:
                        min_num = j
    print(max_num, min_num)


# 把每一帧提取出来放入文件夹
def vedio_picture(root_path, vedio_name):
    video_path = root_path + vedio_name
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 获取视频时长（秒）
    duration = total_frames / fps

    print(f"视频总帧数: {total_frames}")
    print(f"视频帧率: {fps}")
    print(f"视频时长（秒）: {duration}")
    picture_path0 = root_path + "picture_0"
    if not os.path.exists(picture_path0):
        os.makedirs(picture_path0)
    picture_path0 = picture_path0 + '/'
    i = 0
    # 循环读取视频的每一帧
    while True:
        print(1)
        i = i + 1
        # 读取一帧
        ret, frame = cap.read()

        # 如果正确读取帧，ret为True
        if not ret:
            print("无法读取视频帧（可能已到达视频末尾）")
            break

        picture_path = os.path.join(picture_path0, f'{i}.jpg')
        cv2.imwrite(picture_path, frame)
        print(f'Image {picture_path} saved successfully.')


# 图片处理成黑白
def picture_gray(root_path):
    picture_path = root_path + "picture_0"
    files = os.listdir(picture_path)  # 读入文件夹
    num_png = len(files)  # 统计文件夹中的文件个数

    picture_gray_path0 = root_path + "picture_1_grey"
    if not os.path.exists(picture_gray_path0):
        os.makedirs(picture_gray_path0)
    picture_gray_path0 = picture_gray_path0 + '/'
    for i in range(num_png):
        image_path = os.path.join(picture_path, f'{i}.jpg')
        image = cv2.imread(image_path)

        # 检查图片是否成功读取
        if image is None:
            print("无法读取图片")
        else:
            # 将图片转换为灰度图（黑白）
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 保存灰度图
            save_path = os.path.join(picture_gray_path0 + '/',
                                     f'{i}.jpg')  # 替换为你想要保存的路径和文件名
            cv2.imwrite(save_path, gray_image)
            print("灰度图已保存")


# 对灰度图提取边缘
def picture_gray_edge(root_path):
    picture_path = root_path + "picture_0"
    files = os.listdir(picture_path)  # 读入文件夹
    num_png = len(files)  # 统计文件夹中的文件个数

    save_path0 = root_path + "picture_2_edge"
    if not os.path.exists(save_path0):
        os.makedirs(save_path0)
    save_path0 = save_path0 + '/'
    picture_1_grey = root_path + "picture_1_grey/"
    for i in range(num_png):
        image_path = os.path.join(picture_1_grey, f'{i}.jpg')
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 使用cv2.IMREAD_GRAYSCALE直接读取为灰度图

        # 检查图片是否成功读取
        if gray_image is None:
            print("无法读取图片")
        else:
            # 应用Canny边缘检测
            # 第一个参数是输入图像，第二个和第三个参数分别是Canny算法中的低阈值和高阈值
            edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)

            # 保存边缘提取后的图片
            save_path = os.path.join(save_path0, f'{i}.jpg')  # 替换为你想要保存的路径和文件名
            cv2.imwrite(save_path, edges)
            print("边缘提取后的图片已保存")


# 对边缘图像进行膨胀操作
def edge_pengzhang(root_path):
    picture_path = root_path + "picture_0"
    files = os.listdir(picture_path)  # 读入文件夹
    num_png = len(files)  # 统计文件夹中的文件个数

    pengzhang_image_path0 = root_path + "picture_3_edge_pengzhang"
    if not os.path.exists(pengzhang_image_path0):
        os.makedirs(pengzhang_image_path0)
    pengzhang_image_path0 = pengzhang_image_path0 + '/'
    picture_2_edge_path = root_path + 'picture_2_edge/'
    for i in range(num_png):
        edges_image_path = os.path.join(picture_2_edge_path, f'{i}.jpg')
        edges_image = cv2.imread(edges_image_path, cv2.IMREAD_GRAYSCALE)  # 确保以灰度模式读取
        # 检查图片是否成功读取
        if edges_image is None:
            print("无法读取图片")
        else:
            # 定义一个结构元素（kernel），这里使用一个简单的矩形结构
            # 你可以通过调整kernel的大小来改变膨胀的程度
            kernel = np.ones((5, 5), np.uint8)  # 5x5的矩形结构元素

            # 应用膨胀操作
            dilated_image = cv2.dilate(edges_image, kernel, iterations=1)

            # 保存膨胀后的图片
            save_path = os.path.join(pengzhang_image_path0,
                                     f'{i}.jpg')  # 替换为你想要保存的路径和文件名
            cv2.imwrite(save_path, dilated_image)
            print("膨胀后的图片已保存")


# 生成一张底片
def create_picture(create_picture_size):
    new_image = Image.new('RGB', create_picture_size, color='white')

    # 保存图片
    new_image.save('01.jpg')  # 或者使用 'new_image.png' 来保存为PNG格式


# 切割图片到最佳尺寸
def picture_cut(root_path, cut_1, cut_2):
    picture_path = root_path + "picture_0"
    files = os.listdir(picture_path)  # 读入文件夹
    num_png = len(files)  # 统计文件夹中的文件个数
    edge_path = root_path + "picture_2_edge/"

    output_image_path0 = root_path + "picture_3_edge_cut"
    if not os.path.exists(output_image_path0):
        os.makedirs(output_image_path0)
    output_image_path0 = output_image_path0 + '/'
    for i in range(num_png):
        print(i)
        image_path = os.path.join(edge_path, f'{i}.jpg')
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 确保以灰度模式读取
        if image is None:
            print("无法读取图片")
        else:
            image = image[cut_1:cut_2][:]
            output_image_path = os.path.join(output_image_path0, f'{i}.jpg')
            image = Image.fromarray(image, 'L')
            image.save(output_image_path)


# 开始贴图片成为01
def picture_edge_to_01(root_path, create_picture_size, scale):
    # 592*1280    10*15    59*85      590*1275    hengzhe
    # 592*410     10*15    156*72      1559*1080
    # 1560*1080 4*6
    """
    在底图上添加覆盖图，并保存结果。
    base_image_path: 底图的路径
    overlay_image_path: 覆盖图的路径
    output_image_path: 输出图片的路径
    position: 覆盖图在底图上的位置，格式为(x, y)
    """
    picture_path = root_path + "picture_0"
    files = os.listdir(picture_path)  # 读入文件夹
    num_png = len(files)  # 统计文件夹中的文件个数

    base_image = Image.open('01.jpg')

    # 图片所在文件夹
    image_black_folder = os.path.dirname(os.path.dirname(root_path))+'/Abcd_Black/'
    # 读取图片列表并排序
    overlay_image_black = [os.path.join(image_black_folder, img) for img in os.listdir(image_black_folder) if img.endswith(".png") or img.endswith(".jpg")]

    # 图片所在文件夹
    image_white_folder = os.path.dirname(os.path.dirname(root_path))+ '/Abcd_White/'
    # 读取图片列表并排序
    overlay_image_white = [os.path.join(image_white_folder, img) for img in os.listdir(image_white_folder) if img.endswith(".png") or img.endswith(".jpg")]

    overlay_images = {
        'white': [Image.open(path) for path in overlay_image_white],
        'black': [Image.open(path) for path in overlay_image_black]
    }

    size0 = create_picture_size[0]
    size1 = create_picture_size[1]
    scale0 = scale[0]
    scale1 = scale[1]

    output_image_path0 = root_path + "picture_4_01"
    if not os.path.exists(output_image_path0):
        os.makedirs(output_image_path0)
    output_image_path0 = output_image_path0 + '/'
    picture_3_edge_cut_path = root_path + 'picture_3_edge_cut/'
    # 打开底图和覆盖图
    for i in range(num_png):
        print(i)

        image_path = os.path.join(picture_3_edge_cut_path, f'{i}.jpg')
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 确保以灰度模式读取
        if image is None:
            print("无法读取图片")
        else:
            for long in range(0, size1, 15):
                for short in range(0, size0, 10):
                    # print(long/15*scale1,long/15*scale1 + scale1,short/10*scale0,short/10*scale0 + scale0)
                    if image[int(long / 15 * scale1):int(long / 15 * scale1 + scale1),
                       int(short / 10 * scale0):int(short / 10 * scale0 + scale0)].sum() != 0:
                        base_image.paste(random.choice(overlay_images['black']) , (short, long))
                    else:
                        base_image.paste(random.choice(overlay_images['white']), (short, long))
            print(long / 15 * scale1, long / 15 * scale1 + scale1, short / 10 * scale0, short / 10 * scale0 + scale0)
            output_image_path = os.path.join(output_image_path0, f'{i}.jpg')
            base_image.save(output_image_path)

        # 将覆盖图粘贴到底图的指定位置
        # base_image.paste(overlay_image, position, overlay_image)

        # 保存结果图片
        # base_image.save(output_image_path)


def natural_sort_key(s):
    """
    用于自然排序的键函数。
    从字符串中提取出数字和非数字部分，然后将它们用于排序。
    """
    # 使用正则表达式找到所有的数字和非数字部分
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


# 把图片连接成视频
def picture_to_veido(root_path, vedio_name):
    # 图片所在文件夹
    image_folder = root_path + 'picture_4_01/'
    # 读取图片列表并排序
    images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".jpg")]

    # 使用自然排序对文件名进行排序
    images_sorted = sorted(images, key=lambda x: natural_sort_key(os.path.splitext(os.path.basename(x))[0]))

    # 创建视频剪辑
    clip = ImageSequenceClip(images_sorted, fps=30)  # 设置每秒帧数

    # 写入视频文件
    clip.write_videofile(root_path + '/output_' + vedio_name, codec='libx264')


def zh_ch(string):
    return string.encode('gbk').decode(errors='ignore')


# 播放视频
def final_vedio_show(root_path, vedio_name):
    # 替换为你的视频文件路径
    video_path = root_path + '/output_' + vedio_name

    # 使用OpenCV的视频捕获对象
    cap = cv2.VideoCapture(video_path)
    screen_width = 1080  # 假设的屏幕宽度
    screen_height = 800  # 假设的屏幕高度
    while (cap.isOpened()):
        # 逐帧捕获
        ret, frame = cap.read()

        # 如果正确读取帧，ret为True
        if not ret:
            break

        # 调整帧的大小以适应屏幕
        # 注意：这里我们保持宽高比，通过计算缩放因子来实现
        height, width = frame.shape[:2]
        scale_width = screen_width / width
        scale_height = screen_height / height
        scale = min(scale_width, scale_height)
        # 显示结果帧

        # 计算新的宽度和高度
        width = int(width * scale)
        height = int(height * scale)
        # 调整帧的大小
        dim = (width, height)
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # cv2.namedWindow('少帅！！！',cv2.WINDOW_NORMAL)
        cv2.imshow(" ", resized)

        # 按'q'键退出循环
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

            # 释放捕获器和销毁所有OpenCV窗口
    cap.release()
    cv2.destroyAllWindows()


root_path = "E:\A_WORK\LeetCode\Python/zhangxueliang/"
vedio_name = "zhangxueliang.mp4"
# 181 1135
# 155  1175
cut1 = 420
cut2 = 860

# 常规
# 590*410
# 155  1175    1020
create_picture_size = (590*2+1,450*2  )
# 592*1020   10*15     59*58
# 590*1020   10*40
scale = (10/2 , 15/2 )

# 细致
# 592*1020
# create_picture_size = (590*2, 1020*2)
# 592*1020   10*15   590*1020   59*68
# 590*1020   10*40
# scale = ( 10/2,15/2)


vedio_up_long = 1280
vedio_find_edge_short = 1280

# 视频-图片-边缘-膨胀
# vedio_handle1(root_path, vedio_name)

# 找到视频的最佳上下边
#find_up_down_edge(root_path,vedio_up_long,vedio_find_edge_short)

# 边缘图片-01图片   单独调试
#picture_edge_to_01(root_path, create_picture_size, scale)

# 把图片连接成视频   单独调试
# picture_to_veido(root_path, vedio_name)

# 贴图+连接成视频
# vedio_handle2(root_path, vedio_name, create_picture_size, cut1, cut2, scale)

# 播放视频
#show_vedio(root_path, vedio_name)


'''
import cv2


def play_video_in_grayscale(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频文件")
        return

        # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 使用cv2.waitKey的延时来模拟视频播放的帧率
    wait_time = int(1000 / fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

            # 将帧转换为灰度图像
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 显示灰度帧
        cv2.imshow('Grayscale Video', gray_frame)

        # 等待指定的毫秒数或直到任意键盘按键被按下
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

            # 释放VideoCapture对象
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()


# 调用函数
play_video_in_grayscale('E:\A_WORK\LeetCode/zhangxueliang.mp4')
'''
video_path = "E:\A_WORK\LeetCode\Python\zhangxueliang/output_zhangxueliang - 白色人物.mp4"

# 使用OpenCV的视频捕获对象
cap = cv2.VideoCapture(video_path)
screen_width = 1080  # 假设的屏幕宽度
screen_height = 800  # 假设的屏幕高度
while (cap.isOpened()):
    # 逐帧捕获
    ret, frame = cap.read()

    # 如果正确读取帧，ret为True
    if not ret:
        break

    # 调整帧的大小以适应屏幕
    # 注意：这里我们保持宽高比，通过计算缩放因子来实现
    height, width = frame.shape[:2]
    scale_width = screen_width / width
    scale_height = screen_height / height
    scale = min(scale_width, scale_height)
    # 显示结果帧

    # 计算新的宽度和高度
    width = int(width * scale)
    height = int(height * scale)
    # 调整帧的大小
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # cv2.namedWindow('少帅！！！',cv2.WINDOW_NORMAL)
    cv2.imshow("frame", resized)

    # 按'q'键退出循环
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

        # 释放捕获器和销毁所有OpenCV窗口
cap.release()
cv2.destroyAllWindows()