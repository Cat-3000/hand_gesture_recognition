import cv2 as cv
import numpy as np
import time


def get_yCrCb_mask(image):
    # 转YCrCb空间，高斯模糊，均值阈值化,掩膜扣取皮肤区域
    YCrCb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    Y_img, Cr_img, Cb_img = cv.split(YCrCb)
    guassblur = cv.GaussianBlur(Cr_img, (5, 5), 0)
    _, skin_binnary = cv.threshold(guassblur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # Ostu处理
    dst = cv.bitwise_and(image, image, mask=skin_binnary)
    cv.imshow('face_and_hands_got', dst)

    return dst


def get_yCrCb_mask2(image):      # Y亮度：255，Cr:红色分量：255，Cb：蓝色分量：255
    image = cv.GaussianBlur(image, (5, 5), 0)
    # 转YCrCb空间，高斯模糊，均值阈值化,掩膜扣取皮肤区域
    YCrCb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    # 133≤Cr≤173; 77≤Cb≤127

    # 建立掩膜以及椭圆色彩crcb范围区间：
    # lower_hsv = np.array([30, 133, 77])  # 色彩空间高低值
    # upper_hsv = np.array([120, 165, 177])
    lower_hsv = np.array([30, 150, 77])  # 色彩空间高低值 30, 133, 77
    upper_hsv = np.array([220, 175, 177])  # 120, 155, 177
    mask = cv.inRange(YCrCb, lowerb=lower_hsv, upperb=upper_hsv)

    # 开闭运算，内填充以及外部小区域去除
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))  # 选取膨胀核，这里是矩形品质核，大小为（a,b）
    closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    # cv.imshow("closing", closing)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
    # cv.imshow("opening", opening)

    dst = cv.bitwise_and(image, image, mask=opening)
    cv.imshow('face_and_hands_got_crcb', dst)

    return dst


def get_contours_area_max_fourier(image, height, weight):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 进行8领域拉普拉斯边缘检测
    dst = cv.Laplacian(gray, cv.CV_16S, ksize=3)
    laplacian = cv.convertScaleAbs(dst)

    # 发现轮廓
    contours, heriachy = cv.findContours(laplacian, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 对一系列轮廓点坐标按它们围成的区域面积进行排序,sorted()方法，reverse：【True：降序，False：升序】
    contours_sorted = sorted(contours, key=cv.contourArea, reverse=True)    # 列表里面放排序后的多个数组array
    # （法一）保留区域面积最大的轮廓点坐标，就是需要的手部,[0]是除去最外层[],得到第一个三维数组（矩阵）
    # （法二）保留区域面积第二的轮廓点坐标，排除第一大的面部区域，同时保证代码可靠性，只有在皮肤区域大于1的时候才采用第二个区域
    if len(contours_sorted) is 1:
        contour_area_max = contours_sorted[0][:, 0, :]
    else:
        contour_area_max = contours_sorted[0][:, 0, :]    # 三维轮廓点数组[[[1,2]],[[3,4]]]转换为二维坐标点矩阵[[1 2] [3 4]]
    # print(contour_area_max)

    # 建立黑色掩膜
    black_mask = np.zeros([height, weight], np.uint8)

    # 以下包括两部分，第一部分用于显示，无运算意义；
    # 第二部分傅里叶转换并获取一定个数点后返回进行下一步复数转回轮廓点，再reconstruct函数，变回轮廓点坐标，再显示出来
    # 第一部分：将选取的皮肤轮廓（在黑色背景上描白点）；
    # 第二部分：将选取的区域轮廓点转换成傅里叶复数坐标（三维矩阵转一维复数）；

    # 将筛选得到的边界点用白点描绘在黑色掩膜上
    # contours_sorted[0] = [[[1 2]] [[3 4]]], 数组单独显示就是矩阵，多个数组放在一个[],显示时就是列表[a,b,c,d,....]
    if len(contours_sorted) is 1:
        black_mask_nodes = cv.drawContours(black_mask, contours_sorted[0], -1, (255, 255, 255), 1)
        cv.imshow("black_mask_nodes", black_mask_nodes)
    else:
        black_mask_nodes = cv.drawContours(black_mask, contours_sorted[0], -1, (255, 255, 255), 1)
        cv.imshow("black_mask_nodes", black_mask_nodes)

    fourier = get_fourier(contour_area_max)  # 傅里叶中间n个复数项坐标点

    # 返回：傅里叶复数坐标点，目标区域未进行傅里叶转换前的轮廓白点掩膜,得到的手部区域的单独contour（三维矩阵）
    return fourier, black_mask_nodes, contour_area_max


def get_fourier(contour_area_max):
    # 转换坐标为复数坐标
    contours_complex = np.empty(contour_area_max.shape[:-1], dtype=complex)
    # print('contour_area_max.shape[:-1]', contour_area_max.shape[:-1])   # (523,)
    contours_complex.real = contour_area_max[:, 0]  # 横坐标作为实数部分
    contours_complex.imag = contour_area_max[:, 1]  # 纵坐标作为虚数部分

    # 进行傅里叶变换，截取前n个特征因子（复数代表性坐标点）
    fourier_changed = np.fft.fft(contours_complex)  # 进行傅里叶变换

    #  截短傅里叶描述子，得到前n个特征因子并返回
    fourier_nodes_in_use = np.fft.fftshift(fourier_changed)
    # 取中间的n=32项描述子
    fourier_nodes_number = 60
    center_index = int(len(fourier_nodes_in_use) / 2)
    low, high = center_index - int(fourier_nodes_number / 2), center_index + int(fourier_nodes_number / 2)
    fourier_nodes_in_use = fourier_nodes_in_use[low:high]      # 得到中间傅里叶点左右的n/2个点

    fourier_nodes_in_use = np.fft.ifftshift(fourier_nodes_in_use)

    return fourier_nodes_in_use


def reconstruct(img, fourier_nodes_in_use):
    contour_reconstruct = np.fft.ifft(fourier_nodes_in_use)
    contour_reconstruct = np.array([contour_reconstruct.real, contour_reconstruct.imag])
    contour_reconstruct = np.transpose(contour_reconstruct)
    contour_reconstruct = np.expand_dims(contour_reconstruct, axis=1)
    if contour_reconstruct.min() < 0:
        contour_reconstruct -= contour_reconstruct.min()
    contour_reconstruct *= img.shape[0] / contour_reconstruct.max()
    contour_reconstruct = contour_reconstruct.astype(np.int32, copy=False)

    black_np = np.ones(img.shape, np.uint8)  # 创建黑色幕布
    black = cv.drawContours(black_np, contour_reconstruct, -1, (255, 255, 255), 2)  # 绘制白色轮廓
    cv.imshow("contour_reconstruct", black)
    return black


def hu_elements_four_got(contour_area_max, height, weight):
    # 输入变量是手部区域轮廓单独的轮廓点，三维矩阵：[[[1 2]] [[3 4]] ...]
    # 获得质心，并返回以便用在Hu距求取四个特征
    # m:空间矩; mu:中心矩； nu:归一化中心矩
    mm = cv.moments(contour_area_max)  # 获取质心（重心）相关信息，以字典类型保存
    if mm['m00'] != 0:
        cx = mm['m10'] / mm['m00']  # 按重心与字典内储存信息的计算公式得到重心坐标
        cy = mm['m01'] / mm['m00']
    else:
        cx = weight / 2
        cy = height / 2
    # print(cx, cy)     # 质心（重心坐标）

    # 归一化中心矩：归一化后具有尺度不变性
    M1 = mm['nu20'] + mm['nu02']

    M2 = (mm['nu20'] - mm['nu02'])**2 + 4*(mm['nu11']**2)

    M3 = (mm['nu30'] - 3*mm['nu12'])**2 + (3*mm['nu21'] - mm['nu03'])**2

    M4 = (mm['nu30'] + mm['nu12'])**2 + (mm['nu03'] + mm['nu21'])**2

    a5 = (mm['nu30'] - 3*mm['nu12'])*(mm['nu30'] + mm['nu12'])*((mm['nu30'] + mm['nu12'])**2 - 3*(mm['nu21'] + mm['nu30'])**2)
    b5 = (3*mm['nu21'] - mm['nu03'])*(mm['nu21'] + mm['nu03'])*(3*(mm['nu30'] + mm['nu12']**2) - (mm['nu21'] + mm['nu03'])**2)
    M5 = a5 + b5

    a6 = (mm['nu20'] - mm['nu02'])*((mm['nu30'] + mm['nu12'])**2 - (mm['nu21'] + mm['nu03'])**2)
    b6 = 4*mm['nu11']*(mm['nu30'] + mm['nu12'])*(mm['nu21'] + mm['nu03'])
    M6 = a6 + b6

    a7 = (3*mm['nu21'] + mm['nu03'])*(mm['nu30'] + mm['nu12'])*((mm['nu30'] + mm['nu12'])**2 - 3*(mm['nu21'] + mm['nu03'])**2)
    b7 = (3*mm['nu12'] - mm['nu30'])*(mm['nu21'] + mm['nu03'])*(3*(mm['nu30'] + mm['nu12']**2) - (mm['nu21'] + mm['nu03'])**2)
    M7 = a7 + b7

    return M1, M2, M3, M4, M5, M6, M7, cx, cy


def main():
    # 创建一个视频捕捉对象
    cap = cv.VideoCapture(0)  # 0为（笔记本）内置摄像头

    # 计数器
    counter = 0

    # 获取一张背景，用来得知背景尺寸
    ret, frame = cap.read()
    height, weight = frame.shape[:2]
    print('height:{}, weight:{}'.format(height, weight))
    cv.imshow("imput image", frame)  # 将引入的图片在刚刚建立的窗口上面显示出来

    while True:
        # 读帧
        ret, frame = cap.read()
        # 图像翻转
        frame = cv.flip(frame, 2)  # 第二个参数大于0：就表示是沿y轴翻转
        height, weight = frame.shape[:2]

        # 第一步：获得皮肤区域
        face_and_hands = get_yCrCb_mask2(frame)

        # 第二步：获取皮肤区域中的手，并将手轮廓点转成傅里叶轮廓点,以及手部区域的图像质心
        # 获得最大区域轮廓点(只有一个轮廓),并转换成傅里叶点
        # fourier_nodes:一维复数矩阵：[a bj c dj]
        fourier_nodes, black_mask_nodes, contour_area_max = get_contours_area_max_fourier(face_and_hands, height, weight)

        # 将傅里叶点显示出来，以便观看和筛选保存，用于机器学习,并返回转换后的傅里叶子显示图
        black_back = reconstruct(black_mask_nodes, fourier_nodes)     # 将复数转换成轮廓点坐标，再转化二通道图像再显示
        # print(fourier_nodes)
        # print(black_back.shape)

        # 获得轮廓面积和周长的比值
        areas = cv.contourArea(contour_area_max)  # 轮廓面积
        line_long = cv.arcLength(contour_area_max, False)  # 轮廓周长
        if areas < 600:
            continue

        if line_long < 200:
            continue
        else:
            line_compare = areas / line_long

        # 获取几何矩二三阶七个分量的前四个分量（Hu矩）
        M1, M2, M3, M4, M5, M6, M7, cx, cy = hu_elements_four_got(contour_area_max, height, weight)

        # 保存图片,包括扣取区域图，区域轮廓图，以及特征参数
        key = cv.waitKey(10)
        if key is 9:
            # 将特征参数写在轮廓图上，包括五个特征元素以及一个质心坐标
            # 设置一些常用的一些参数
            # 显示的字体 大小 初始位置等
            font = cv.FONT_HERSHEY_SIMPLEX  # 正常大小无衬线字体
            size = 0.5
            fx = 10
            fy = 10
            fh = 18
            # 将单通道二值图合并成三通道
            merge = cv.merge([black_mask_nodes, black_mask_nodes, black_mask_nodes])
            cv.putText(merge, "M1：{}".format(M1), (fx, fy), font, size, (0, 255, 0))  # 标注字体
            cv.putText(merge, "M2：{}".format(M2), (fx, fy + fh), font, size, (0, 255, 0))  # 标注字体
            cv.putText(merge, "M3：{}".format(M3), (fx, fy + 2*fh), font, size, (0, 255, 0))  # 标注字体
            cv.putText(merge, "M4：{}".format(M4), (fx, fy + 3*fh), font, size, (0, 255, 0))  # 标注字体
            cv.putText(merge, "M5：{}".format(M5), (fx, fy + 4*fh), font, size, (0, 255, 0))  # 标注字体
            cv.putText(merge, "M6：{}".format(M6), (fx, fy + 5*fh), font, size, (0, 255, 0))  # 标注字体
            cv.putText(merge, "M7：{}".format(M7), (fx, fy + 6*fh), font, size, (0, 255, 0))  # 标注字体
            cv.putText(merge, "line_compare：{}".format(line_compare), (fx, fy+4*fh), font, size, (0, 255, 0))
            cv.putText(merge, "(cx,cy)：{},{}".format(cx, cy), (fx, fy+5*fh), font, size, (0, 255, 0))  # 标注字体

            # 在原图质心画点
            # 根据重心坐标画圆，3是半径，-1是填充模式，颜色为绿色
            cv.circle(merge, (np.int(cx), np.int(cy)), 3, (0, 255, 0), -1)

            # 保存将信息添加以后的轮廓图片到选择的路径
            path = "D:/opencv_photo/project_saved/merge/"
            path2 = "D:/opencv_photo/project_saved/frame/"
            name = 'result' + str(counter)  # 给录制的手势命名
            # print("Saving img: ", name)
            cv.imwrite(path + name + '.png', merge)  # 写入文件
            cv.imwrite(path2 + name + '.png', frame)  # 写入文件
            time.sleep(0.05)
            counter += 1
            if counter >= 500:
                counter = 0

            # 将保存的图片显示出来
            cv.imshow("black_mask_nodes_Tab", merge)
            cv.imshow("frame_Tab", frame)

            # 保存数据到文件
            # ......
            # print('counter{}:'.format(counter), M1, M2, M3, M4, M5, M6, M7, line_compare, cx, cy)
            print('counter{}:'.format(counter), M1, M2, M3, M4, M5, M6, M7, line_compare)

        # 展示处理之后的视频帧
        cv.imshow('frame', frame)

        c = cv.waitKey(10)
        if c is 27:
            break


if __name__ == "__main__":
    print("---------stare to get a picture-----------")

    cv.namedWindow("frame", cv.WINDOW_AUTOSIZE)   # 建立一个窗口，此时还没有要显示的东西

    t1 = cv.getTickCount()     # getTickCount():返回系统启动到当前所经历计时周期数

    main()

    t2 = cv.getTickCount()
    time = (t2 - t1)/cv.getTickFrequency()*1000   # getTickFrequency()：返回CPU的频率
    print("use time:{}ms".format(time))

    cv.waitKey(0)      # 值为0，延时无先长，直到有按键按下才继续执行后续代码，值大于零时延时对应值时间，单位ms
    cv.destroyAllWindows()    # 释放内存（窗口）

