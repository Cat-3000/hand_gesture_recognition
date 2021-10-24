## hand_gesture_recognition
使用opencv提取手势数据，使用神经网络训练出想要的手势模型，最后将手势模型加载到手势识别程序完成手势命令的获取；
工具：opencv3
演示视频地址：https://www.bilibili.com/video/BV1P3411k7d5/

# 写在前面：
项目内容是我大三时参加比赛一边学习一边完成的，有很多不足之处但目前还没有优化

## 程序运行须知：
# 第一步：获取手势Hu矩数据和几个其他数据
运行过程：修改路径，修改色彩空间找到适合自己环境的hsv值，运行程序，对合适的时机按下Tab键（按下获取，松开不获取），获取足够数据量后按Esc两次退出程序
修改：打开get_test.py，修改获取手势图片存储路径：
	path = "D:/opencv_photo/project_saved/merge/"（存获取时照片）
	path2 = "D:/opencv_photo/project_saved/frame/"（存获取时手势区域）
	修改为自己的文件目录路径

	get_yCrCb_mask2（）函数中：
	lower_hsv = np.array([30, 150, 77])  # 色彩空间高低值 30, 133, 77
	upper_hsv = np.array([220, 175, 177])  # 120, 155, 177
	修改色彩空间范围，最好背景干净与肤色区别明显，同时光线良好
	
	开始运行
数据处理：将打印数据存入自己建立的各个手势文档（一个手势在500个左右，因为手势包括两手，训练时需要等量原则）

# 第二步：数据处理
	这里方法很多，可以手动也可以代码自动，就不提供代码了
	将获取的数据按等量原则放入hands_elements_data_all.csv表中，同时做好标签如（提供的hands_elements_data_all.csv作为参考）：

	INDUS	CHAS	NOX	RM	AGE	DIS	ZERO	ONE	TWO	THREE	FOUR	FIVE	FZ	FACE
	4.92E-04	4.11E-05	1.16E-08	3.24E-06	-1.11E-06	30.16510035	1	0	0	0	0	0	0	0
	5.43E-04	4.62E-05	-1.08E-07	3.53E-06	-1.32E-06	29.66818503	1	0	0	0	0	0	0	0
	5.00E-04	4.80E-05	-1.12E-07	3.76E-06	-1.02E-06	29.87976689	1	0	0	0	0	0	0	0

	将每个要训练的手势模型作为正样本，其他模型为负样本，正负等量原则将数据放入hands_elements_data.csv表中，做好标签如：
	1.30E-02	6.03E-04	1.36E-05	3.79E-07	-2.51E-07	-3.25E-07	35.31242498	0	0	0	0	0	0	0	1
	1.14E-02	7.50E-04	5.80E-05	-2.10E-07	5.80E-06	-1.69E-08	37.11883482	0	0	0	0	0	0	0	1
	1.96E-02	2.27E-03	5.71E-05	2.15E-06	4.41E-06	5.82E-06	25.00462225	0	0	0	0	0	0	1	0



# 第三步：模型训练
运行过程：打开hands_learning.py,修改导入训练数据表所在路径：
	视情况修改训练超参数，比如数据量不同时：
	learning_rate = 1e-1
	learning_times = 100000  # 学习轮数
	date_all = 1600          # 训练集数据个数
	date_detect = 360       # 测试集数据个数

	# 导入数据
	hands_elements_all = pandas.read_csv("D:/opencv_photo/project_saved//hands_elements_data_all.csv")
	hands_elements = pandas.read_csv("D:/opencv_photo/project_saved//hands_elements_data.csv") 
	
	修改训练手势目标模型索引键('ZERO'代表一种手势）：
	date_used_by_y = hands_elements['ZERO']
	
	开始运行（这里最开始的极限数据记下来，后面要用）
将打印出的符合要求的参数值记录下

# 第四步：使用模型
运行过程：打开result.py，修改参数
	拍摄手势对应照片（修改大小），包括各个手势，以及脸，几个手势，以及一张空白图片；
	放于路径如D:/opencv_photo/project_saved/image_show/no_hand.png
	同时将程序中全部图片存放路径都改为你设置的路径

	修改模型内容：    
	# 一个判断轮的帧数
	modoul_all = ('zero', 'one', 'five', 'five_zero', 'face')   # 包括训练的手势，我的模型里面有一个是脸部模型

	修改模型数量及参数：
	将def modoul_eight(modoul_select)函数下对应的模型名称和参数对应填入

	将moudle_detect_result_eight（）函数的模型参数极限值进行修改，这里在模型训练时运行程序一开始就会打印出，直接移过来

	修改def get_yCrCb_mask2(image)函数色彩空间，与手势数据获取时得到的色彩空间相同

	运行程序，最多会出现三个窗口，代表脸部，左手，右手（按照提取皮肤区域大小排序前三出现），不对应锁定窗口，
	
# 结果：
如果正常，识别到对应手势或脸部或无时，各个窗口会对应弹出识别结果图片

# 声明：
项目为自制，上传仅用于学习交流


	



