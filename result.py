import cv2 as cv
import numpy as np


def get_yCrCb_mask(image):
    # 转YCrCb空间，高斯模糊，均值阈值化,掩膜扣取皮肤区域
    YCrCb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    Y_img, Cr_img, Cb_img = cv.split(YCrCb)
    guassblur = cv.GaussianBlur(Cr_img, (5, 5), 0)
    _, skin_binnary = cv.threshold(guassblur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # Ostu处理
    dst = cv.bitwise_and(image, image, mask=skin_binnary)
    cv.imshow('face_and_hands_got', dst)

    return dst


def get_yCrCb_mask2(image):  # Y亮度：255，Cr:红色分量：255，Cb：蓝色分量：255
    image = cv.GaussianBlur(image, (5, 5), 0)
    # 转YCrCb空间，高斯模糊，均值阈值化,掩膜扣取皮肤区域
    YCrCb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    # 133≤Cr≤173; 77≤Cb≤127

    # 建立掩膜以及椭圆色彩crcb范围区间：
    lower_hsv = np.array([30, 150, 77])  # 色彩空间高低值 30, 133, 77 ，：30,150,77            30,140，77
    upper_hsv = np.array([220, 175, 177])  # 120, 155, 177           ，：220，175,177         220,175,177
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
    contours_sorted = sorted(contours, key=cv.contourArea, reverse=True)  # 列表里面放排序后的多个数组array
    # （法一）保留区域面积最大的轮廓点坐标，就是需要的手部,[0]是除去最外层[],得到第一个三维数组（矩阵）
    # （法二）保留区域面积第二的轮廓点坐标，排除第一大的面部区域，同时保证代码可靠性，只有在皮肤区域大于1的时候才采用第二个区域
    # print("len(contours_sorted)", len(contours_sorted))

    if len(contours_sorted) == 0:
        return 0, 0, 0, 0, 0, 0, 0

    if len(contours_sorted) == 1:
        contour_area_max_zero = contours_sorted[0][:, 0, :]
    elif len(contours_sorted) == 2:
        contour_area_max_zero = contours_sorted[0][:, 0, :]  # 三维轮廓点数组[[[1,2]],[[3,4]]]转换为二维坐标点矩阵[[1 2] [3 4]]
        area_one = contours_sorted[1][:, 0, :]
    elif len(contours_sorted) >= 3:
        contour_area_max_zero = contours_sorted[0][:, 0, :]
        area_one = contours_sorted[1][:, 0, :]
        area_two = contours_sorted[2][:, 0, :]
    # print(contour_area_max)

    # 建立黑色掩膜,三张不同地址位置
    black_mask = np.zeros([height, weight], np.uint8)
    black_mask_zero = black_mask.copy()
    black_mask_one = black_mask.copy()
    black_mask_two = black_mask.copy()

    # 将筛选得到的边界点用白点描绘在黑色掩膜上
    # contours_sorted[0] = [[[1 2]] [[3 4]]], 数组单独显示就是矩阵，多个数组放在一个[],显示时就是列表[a,b,c,d,....]
    if len(contours_sorted) == 1:
        black_mask_nodes_zero = cv.drawContours(black_mask_zero, contours_sorted[0], -1, (255, 255, 255), 1)
        cv.imshow("nodes_zero", black_mask_nodes_zero)
    elif len(contours_sorted) == 2:
        black_mask_nodes_zero = cv.drawContours(black_mask_zero, contours_sorted[0], -1, (255, 255, 255), 1)
        mask_one = cv.drawContours(black_mask_one, contours_sorted[1], -1, (255, 255, 255), 1)
        cv.imshow("nodes_zero", black_mask_nodes_zero)
        cv.imshow("nodes_one", mask_one)
    elif len(contours_sorted) >= 3:
        black_mask_nodes_zero = cv.drawContours(black_mask_zero, contours_sorted[0], -1, (255, 255, 255), 1)
        mask_one = cv.drawContours(black_mask_one, contours_sorted[1], -1, (255, 255, 255), 1)
        mask_two = cv.drawContours(black_mask_two, contours_sorted[2], -1, (255, 255, 255), 1)
        cv.imshow("nodes_zero", black_mask_nodes_zero)
        cv.imshow("nodes_one", mask_one)
        cv.imshow("nodes_two", mask_two)

    if len(contours_sorted) is 1:
        return black_mask_nodes_zero, contour_area_max_zero, 0, 0, 0, 0, 1
    elif len(contours_sorted) is 2:
        return black_mask_nodes_zero, contour_area_max_zero, mask_one, area_one, 0, 0, 2
    elif len(contours_sorted) >= 3:
        return black_mask_nodes_zero, contour_area_max_zero, mask_one, area_one, mask_two, area_two, 3


def hu_elements_seven_got(contour_area_max, height, weight):
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

    M2 = (mm['nu20'] - mm['nu02']) ** 2 + 4 * (mm['nu11'] ** 2)

    M3 = (mm['nu30'] - 3 * mm['nu12']) ** 2 + (3 * mm['nu21'] - mm['nu03']) ** 2

    M4 = (mm['nu30'] + mm['nu12']) ** 2 + (mm['nu03'] + mm['nu21']) ** 2

    a5 = (mm['nu30'] - 3 * mm['nu12']) * (mm['nu30'] + mm['nu12']) * (
            (mm['nu30'] + mm['nu12']) ** 2 - 3 * (mm['nu21'] + mm['nu30']) ** 2)
    b5 = (3 * mm['nu21'] - mm['nu03']) * (mm['nu21'] + mm['nu03']) * (
            3 * (mm['nu30'] + mm['nu12'] ** 2) - (mm['nu21'] + mm['nu03']) ** 2)
    M5 = a5 + b5

    a6 = (mm['nu20'] - mm['nu02']) * ((mm['nu30'] + mm['nu12']) ** 2 - (mm['nu21'] + mm['nu03']) ** 2)
    b6 = 4 * mm['nu11'] * (mm['nu30'] + mm['nu12']) * (mm['nu21'] + mm['nu03'])
    M6 = a6 + b6

    a7 = (3 * mm['nu21'] + mm['nu03']) * (mm['nu30'] + mm['nu12']) * (
            (mm['nu30'] + mm['nu12']) ** 2 - 3 * (mm['nu21'] + mm['nu03']) ** 2)
    b7 = (3 * mm['nu12'] - mm['nu30']) * (mm['nu21'] + mm['nu03']) * (
            3 * (mm['nu30'] + mm['nu12'] ** 2) - (mm['nu21'] + mm['nu03']) ** 2)
    M7 = a7 + b7

    return M1, M2, M3, M4, M5, M6, M7, cx, cy


def modoul_eight(modoul_select):
    if modoul_select is 'zero':
        # k1:-167.4208330892378, k2:136.96566655810832, k3:138.6380972605112
        # k4:-147.6778022197367, k5:-68.32634600216001, k6:-141.9571682046833
        # k7:-241.7356806033519, k8:-12.815337711070475, w:-127.21459280676439
        # k = [-167.4208330892378, 136.96566655810832, 138.6380972605112, -147.6778022197367,
        #      -68.32634600216001, -141.9571682046833, -241.7356806033519, -12.815337711070475]
        # w = [-127.21459280676439]

        # k1:-124.05734475693754, k2:85.9528114658731, k3:110.37923222005031
        # k4:-41.38456897505413, k5:28.61659924518532, k6:-45.48200370059477
        # k7:-165.19217135208376, k8:-6.342577049473227, w:-70.02992775115159
        k = [-124.05734475693754, 85.9528114658731, 110.37923222005031, -41.38456897505413,
             28.61659924518532, -45.48200370059477, -165.19217135208376, -6.342577049473227]
        w = [-70.02992775115159]

    if modoul_select is 'one':
        # k1:-64.7919652202576, k2:70.71983535263175, k3:20.522048940130315
        # k4:142.85724930510716, k5:23.826722640195722, k6:-201.89933020197762
        # k7:9.660697455322525, k8:-4.502786310697994, w:2.2951824387608206
        k = [-64.7919652202576, 70.71983535263175, 20.522048940130315, 142.85724930510716,
             23.826722640195722, -201.89933020197762, 9.660697455322525, -4.502786310697994]
        w = [2.2951824387608206]

    if modoul_select is 'five':
        # k1:9.30261151853744, k2:-62.716852298603996, k3:-30.90162129703201
        # k4:-2.848421501749639, k5:74.50488950376445, k6:-1.8627117673044042
        # k7:32.73002748817182, k8:-36.767993864831105, w:13.643044774964627
        # k = [7.908870817671702, -60.09631228241662, -65.25706699537679, -5.884275199501333,
        #      45.37393347461258, -4.461858864035195, 15.239893963664475, -37.90345586234362]
        # w = [0.6751818438806183]

        # k1:11.08802318974459, k2:-64.82067958904126, k3:-74.95769706012756
        # k4:-13.748471804048, k5:51.07304855210955, k6:-7.633360111267454
        # k7:17.85288514881367, k8:-38.05293459124017, w:2.7506628124706882
        k = [11.08802318974459, -64.82067958904126, -74.95769706012756, -13.748471804048,
             51.07304855210955, -7.633360111267454, 17.85288514881367, -38.05293459124017]
        w = [2.7506628124706882]

    if modoul_select is 'five_zero':
        # k1:-27.9353673233159, k2:45.603498899473905, k3:100.84148125719777
        # k4:-199.36363062516588, k5:59.46708316827832, k6:-88.4568007497318
        # k7:47.04003986918764, k8:2.020027319542766, w:34.43476791148194
        k = [-27.9353673233159, 45.603498899473905, 100.84148125719777, -199.36363062516588,
             59.46708316827832, -88.4568007497318, 47.04003986918764, 2.020027319542766]
        w = [34.43476791148194]

    if modoul_select is 'face':
        # k1:-0.6381707407388115, k2:2.5151416794683996, k3:-71.83025032016374
        # k4:-127.88714876952272, k5:7.896844727246807, k6:-59.72687945947059
        # k7:4.857338347222226, k8:9.345859020821361, w:3.496291837315175
        # k = [-0.6381707407388115, 2.5151416794683996, -71.83025032016374, -127.88714876952272,
        #      7.896844727246807, -59.72687945947059, 4.857338347222226, 9.345859020821361]
        # w = [3.496291837315175]

        # k1:18.538202103415124, k2:-26.050369153494326, k3:-74.52453980022342
        # k4:-301.91571083317695, k5:-52.80197070741301, k6:-110.45596280896515
        # k7:16.512587174444555, k8:15.200610763139546, w:-2.144556712227466
        k = [18.538202103415124, -26.050369153494326, -74.52453980022342, -301.91571083317695,
             -52.80197070741301, -110.45596280896515, 16.512587174444555, 15.200610763139546]
        w = [-2.144556712227466]

        # k1:18.068293719765304, k2:-25.631779667631996, k3:-74.43920295880332
        # k4:-292.32150487664995, k5:-51.04743200809823, k6:-106.75069304103279
        # k7:17.23552975303277, k8:15.08813070264382, w:-1.4900948259329498
        # k = [18.068293719765304, -25.631779667631996, -74.43920295880332, -292.32150487664995,
        #      -51.04743200809823, -106.75069304103279, 17.23552975303277, 15.08813070264382]
        # w = [-1.4900948259329498]

    k1 = k[0]
    k2 = k[1]
    k3 = k[2]
    k4 = k[3]
    k5 = k[4]
    k6 = k[5]
    k7 = k[6]
    k8 = k[7]
    w = w[0]

    return k1, k2, k3, k4, k5, k6, k7, k8, w


def moudle_detect_result_eight(m1, m2, m3, m4, m5, m6, m7, line_compare, k1, k2, k3, k4, k5, k6, k7, k8, w, name):
    x1_max = 0.429
    x1_min = 0.16399999999999998
    x2_max = 0.11900000000000001
    x2_min = 0.000261
    x3_max = 0.0299
    x3_min = 1.27e-10
    x4_max = 0.0185
    x4_min = 9.620000000000001e-09
    x5_max = 0.000278
    x5_min = -7.08e-05
    x6_max = 0.00619
    x6_min = -0.000103
    x7_max = 0.000134
    x7_min = -8.97e-05
    x8_max = 53.87557145
    x8_min = 8.002980413

    node_x1 = (m1 - x1_min) / (x1_max - x1_min)
    node_x2 = (m2 - x2_min) / (x2_max - x2_min)
    node_x3 = (m3 - x3_min) / (x3_max - x3_min)
    node_x4 = (m4 - x4_min) / (x4_max - x4_min)
    node_x5 = (m5 - x5_min) / (x5_max - x5_min)
    node_x6 = (m6 - x6_min) / (x6_max - x6_min)
    node_x7 = (m7 - x7_min) / (x7_max - x7_min)
    node_x8 = (line_compare - x8_min) / (x8_max - x8_min)

    # 计算过程
    linear = node_x1 * k1 + node_x2 * k2 + node_x3 * k3 + node_x4 * k4 + node_x5 * k5 - w
    linear = linear + node_x6 * k6 + node_x7 * k7 + node_x8 * k8
    sigmoid_value = sigmoid(linear)

    return sigmoid_value


def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))  # Define sigmoid function
    if sig >= 1:
        sig = 0.999999999999999
    if sig <= 0:
        sig = 0.000000000000001
    return sig

    # if x >= 0:      # 对sigmoid函数的优化，避免了出现极大的数据溢出
    #     return 1.0/(1+np.exp(-x))
    # else:
    #     return np.exp(x)/(1+np.exp(x))


def image_show(image_name, rate, src2, window, cx, cy):
    global one_circle_image_counter, finally_result, count_the_window_each_time_it_appears
    path = "D:/opencv_photo/project_saved/image_show/"
    name = image_name
    src = cv.imread(path + name + '.png')

    if finally_result['counter'][0] >= 1000:
        finally_result['counter'][0] = finally_result['counter'][1] = 0

    # 统计每次三个窗口有几个得到检测结果，并将结果记录下来
    for _ in count_the_window_each_time_it_appears:       # 检测每轮有多少个窗口得到未过滤的皮肤区域
        if _ is window:
            finally_result['counter'][1] += 1
            count_the_window_each_time_it_appears = []  # 清零，进入新一次窗口出现统计

    count_the_window_each_time_it_appears.append(window)

    if finally_result['counter'][0] == finally_result['counter'][1]:  # 还未有窗口在三个窗口出现完之前重复出现
        if rate >= (one_circle_image_counter // 2):
            finally_result[window] = [name, (int(cx), int(cy))]
    else:  # 检测得到结果的窗口没有达到三个
        finally_result['counter'][0] = finally_result['counter'][1]
        print('finally_rusult:', finally_result)
        finally_result['zero'] = ['no', (0, 0)]
        finally_result['one'] = ['no', (0, 0)]
        finally_result['two'] = ['no', (0, 0)]
        finally_result[window] = [name, (int(cx), int(cy))]

    if rate >= (one_circle_image_counter//2):  # 一轮10张图片，目标概率大于0.95并且出现超过5次的显示相应图片
        cv.imshow("detect_resulu_" + window, src)
    else:
        cv.imshow("detect_resulu_" + window, src2)


def contours_max_zero(contour_area_max, height, weight):
    global one_circle_image_number_zero
    global one_circle_times_zero
    global modoul_all
    global one_circle_image_counter
    src2 = cv.imread("D:/opencv_photo/project_saved/image_show/no.png")
    # 获得轮廓面积和周长的比值
    areas = cv.contourArea(contour_area_max)  # 轮廓面积
    line_long = cv.arcLength(contour_area_max, False)  # 轮廓周长

    # 最小或最大面积
    area_min = height * weight / 40
    area_max = height * weight

    if areas < area_min or areas > area_max:
        return 0

    if line_long < 200:
        return 0
    else:
        line_compare = areas / line_long
    # 获取几何矩二三阶七个分量（Hu矩）
    M1, M2, M3, M4, M5, M6, M7, cx, cy = hu_elements_seven_got(contour_area_max, height, weight)

    # 利用训练得到的神经网络进行判断
    # modoul_all = ('zero', 'one', 'two', 'three', 'four', 'five', 'five_zero', 'face')

    # 建立初值为零的字典，用于计算一轮判断各种手势出现次数
    if one_circle_image_number_zero is 1:
        one_circle_times_zero = dict.fromkeys(modoul_all, 0)
    modoul_all_value = {}  # 建立字典用于储存各个模型的计算结果：概率

    # 检测所有手势
    for _ in modoul_all:
        k1, k2, k3, k4, k5, k6, k7, k8, w = modoul_eight(_)
        detect_result = moudle_detect_result_eight(M1, M2, M3, M4, M5, M6, M7, line_compare, k1,
                                                   k2, k3, k4, k5, k6, k7, k8, w, _)
        modoul_all_value[_] = detect_result

    # 获得符号阈值之上且匹配度最高的手势
    name_ = max(modoul_all_value, key=modoul_all_value.get)
    rate_ = modoul_all_value[name_]
    if rate_ >= 0.95:
        one_circle_image_number_zero += 1
        if one_circle_image_number_zero >= one_circle_image_counter:
            one_circle_image_number_zero = 1

        # 将匹配度最高并且高于0.95的手势计数加一
        one_circle_times_zero[name_] += 1

    # 获得匹配次数最多的手势
    name = max(one_circle_times_zero, key=one_circle_times_zero.get)
    rate = one_circle_times_zero[name]

    if one_circle_image_number_zero == (one_circle_image_counter-1):
        # 显示对应结果
        image_show(name, rate, src2, 'zero', cx, cy)  # detect_result:是概率
        # print('zero', one_circle_times_zero)


def contours_max_one(contour_area_max, height, weight):
    global one_circle_image_number_one
    global one_circle_times_one
    global modoul_all
    global one_circle_image_counter
    src2 = cv.imread("D:/opencv_photo/project_saved/image_show/no.png")
    # 获得轮廓面积和周长的比值
    areas = cv.contourArea(contour_area_max)  # 轮廓面积
    line_long = cv.arcLength(contour_area_max, False)  # 轮廓周长

    # 最小或最大面积
    area_min = height * weight / 40
    area_max = height * weight

    if areas < area_min or areas > area_max:
        return 0

    if line_long < 200:
        return 0
    else:
        line_compare = areas / line_long
    # 获取几何矩二三阶七个分量（Hu矩）
    M1, M2, M3, M4, M5, M6, M7, cx, cy = hu_elements_seven_got(contour_area_max, height, weight)

    # 利用训练得到的神经网络进行判断
    # modoul_all = ('zero', 'one', 'two', 'three', 'four', 'five', 'five_zero', 'face')

    # 建立初值为零的字典，用于计算一轮判断各种手势出现次数
    if one_circle_image_number_one is 1:
        one_circle_times_one = dict.fromkeys(modoul_all, 0)
    modoul_all_value = {}  # 建立字典用于储存各个模型的计算结果：概率

    # 检测所有手势
    for _ in modoul_all:
        k1, k2, k3, k4, k5, k6, k7, k8, w = modoul_eight(_)
        detect_result = moudle_detect_result_eight(M1, M2, M3, M4, M5, M6, M7, line_compare, k1,
                                                   k2, k3, k4, k5, k6, k7, k8, w, _)
        modoul_all_value[_] = detect_result

    # 获得符号阈值之上且匹配度最高的手势
    name_ = max(modoul_all_value, key=modoul_all_value.get)
    rate_ = modoul_all_value[name_]
    if rate_ >= 0.95:
        one_circle_image_number_one += 1
        if one_circle_image_number_one >= one_circle_image_counter:
            one_circle_image_number_one = 1

        # 将匹配度最高并且高于0.95的手势计数加一
        one_circle_times_one[name_] += 1

    # 获得匹配次数最多的手势
    name = max(one_circle_times_one, key=one_circle_times_one.get)
    rate = one_circle_times_one[name]

    if one_circle_image_number_one == (one_circle_image_counter-1):
        # 显示对应结果
        image_show(name, rate, src2, 'one', cx, cy)  # detect_result:是概率
        # print('one', one_circle_times_one)


def contours_max_two(contour_area_max, height, weight):
    global one_circle_image_number_two
    global one_circle_times_two
    global modoul_all
    global one_circle_image_counter
    src2 = cv.imread("D:/opencv_photo/project_saved/image_show/no.png")
    # 获得轮廓面积和周长的比值
    areas = cv.contourArea(contour_area_max)  # 轮廓面积
    line_long = cv.arcLength(contour_area_max, False)  # 轮廓周长

    # 最小或最大面积
    area_min = height * weight / 40
    area_max = height * weight

    if areas < area_min or areas > area_max:  # <600
        return 0

    if line_long < 200:
        return 0
    else:
        line_compare = areas / line_long
    # 获取几何矩二三阶七个分量（Hu矩）
    M1, M2, M3, M4, M5, M6, M7, cx, cy = hu_elements_seven_got(contour_area_max, height, weight)

    # 利用训练得到的神经网络进行判断
    # modoul_all = ('zero', 'one', 'two', 'three', 'four', 'five', 'five_zero', 'face')

    # 建立初值为零的字典，用于计算一轮判断各种手势出现次数
    if one_circle_image_number_two is 1:
        one_circle_times_two = dict.fromkeys(modoul_all, 0)
    modoul_all_value = {}  # 建立字典用于储存各个模型的计算结果：概率

    # 检测所有手势
    for _ in modoul_all:
        k1, k2, k3, k4, k5, k6, k7, k8, w = modoul_eight(_)
        detect_result = moudle_detect_result_eight(M1, M2, M3, M4, M5, M6, M7, line_compare, k1,
                                                   k2, k3, k4, k5, k6, k7, k8, w, _)
        modoul_all_value[_] = detect_result

    # 获得符号阈值之上且匹配度最高的手势
    name_ = max(modoul_all_value, key=modoul_all_value.get)
    rate_ = modoul_all_value[name_]
    if rate_ >= 0.95:
        one_circle_image_number_two += 1
        if one_circle_image_number_two >= one_circle_image_counter:
            one_circle_image_number_two = 1

        # 将匹配度最高并且高于0.95的手势计数加一
        one_circle_times_two[name_] += 1

    # 获得匹配次数最多的手势
    name = max(one_circle_times_two, key=one_circle_times_two.get)
    rate = one_circle_times_two[name]

    if one_circle_image_number_two == (one_circle_image_counter-1):
        # 显示对应结果
        image_show(name, rate, src2, 'two', cx, cy)  # detect_result:是概率
        # print('two', one_circle_times_two)


def main():
    # 创建一个视频捕捉对象
    cap = cv.VideoCapture(0)  # 0为（笔记本）内置摄像头, "D:/opencv_photo/video/get.mp4"

    # 获取一张背景，用来得知背景尺寸
    ret, frame = cap.read()
    height, weight = frame.shape[:2]
    print('height:{}, weight:{}'.format(height, weight))
    # cv.imshow("imput image", frame)  # 将引入的图片在刚刚建立的窗口上面显示出来

    src2 = cv.imread("D:/opencv_photo/project_saved/image_show/no_hand.png")

    while True:
        # 读帧
        ret, frame = cap.read()
        # 图像翻转
        frame = cv.flip(frame, 2)  # 第二个参数大于0：就表示是沿y轴翻转
        height, weight = frame.shape[:2]

        # 第一步：获得皮肤区域
        face_and_hands = get_yCrCb_mask2(frame)

        # 第二步：返回两个参数：一是轮廓图，而是轮廓点
        black_mask_nodes, contour_area_max, mask_one, area_one, mask_two, area_two, len_ = get_contours_area_max_fourier(
            face_and_hands, height, weight)

        # 如果没有任何皮肤区域信息，读取下一张图片
        # if black_mask_nodes is 0 and contour_area_max is 0:
        if len_ is 0:
            continue

        # 有皮肤区域时，第一个皮肤区域
        # if black_mask_nodes is not 0 and contour_area_max is not 0:
        if len_ is 1:
            # cv.imshow("nodes_zero", black_mask_nodes)
            contours_max_zero(contour_area_max, height, weight)
        # 皮肤区域为二时
        # if mask_one is not 0 and area_one is not 0:
        if len_ is 2:
            # cv.imshow("nodes_one", mask_one)
            contours_max_zero(contour_area_max, height, weight)
            contours_max_one(area_one, height, weight)
        # 皮肤区域大于二时
        # if mask_two is not 0 and area_two is not 0:
        if len_ >= 3:
            # cv.imshow("nodes_two", mask_two)
            contours_max_zero(contour_area_max, height, weight)
            contours_max_one(area_one, height, weight)
            contours_max_two(area_two, height, weight)

        # 展示处理之后的视频帧
        cv.imshow('frame', frame)

        c = cv.waitKey(10)
        if c is 27:
            break


if __name__ == "__main__":
    print("---------stare to get a picture-----------")

    cv.namedWindow("frame", cv.WINDOW_AUTOSIZE)  # 建立一个窗口，此时还没有要显示的东西

    t1 = cv.getTickCount()  # getTickCount():返回系统启动到当前所经历计时周期数

    # 一个判断轮的帧数
    modoul_all = ('zero', 'one', 'five', 'five_zero', 'face')
    one_circle_image_number_zero = 1
    one_circle_times_zero = dict.fromkeys(modoul_all, 0)
    one_circle_image_number_one = 1
    one_circle_times_one = dict.fromkeys(modoul_all, 0)
    one_circle_image_number_two = 1
    one_circle_times_two = dict.fromkeys(modoul_all, 0)
    one_circle_image_counter = 6    # 1+10=11

    # 记录最终结果
    count_the_window_each_time_it_appears = []
    finally_result = {'zero': ['no', (0, 0)], 'one': ['no', (0, 0)], 'two': ['no', (0, 0)], 'counter': [0, 0]}

    main()

    t2 = cv.getTickCount()
    time = (t2 - t1) / cv.getTickFrequency() * 1000  # getTickFrequency()：返回CPU的频率
    print("use time:{}ms".format(time))

    cv.waitKey(0)  # 值为0，延时无先长，直到有按键按下才继续执行后续代码，值大于零时延时对应值时间，单位ms
    cv.destroyAllWindows()  # 释放内存（窗口）

