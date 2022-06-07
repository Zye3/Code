# Code
###数据转化和分割：

   		1.txt 文件生产 每张图片对应的.pts 文件
		2.将文件夹数据，按9：1 分成10份，用于交叉验证.


###SDU模型的mxnet框架数据集制作（4）/home/ye/Desktop/Model/insightface-master/alignment/src/data/

	1.使用标注关键点截取固定尺寸的人脸框，并转换坐标 （可以参考DANpreprocessing.py中的getAffine来转换坐标并
          且截取人脸框，训练集固定尺寸为112x112，测试和验证集固定尺寸为384x384)
	  输入:原始人脸和坐标
	  输出:截取人脸框和转换后坐标

	2.制作.lst文件（index，lable，path）       /dir2lable.py:要求图片格式为.jpg;
	  输出:截取人脸框和转换后坐标
	  输出:.lst文件，index，label，path

	3.制作.rec文件 (输入.lst)和.idx文件        /dir2rec.py:要求数据集图片尺寸大于112x112;
	  输入:.lst文件
	  输出：.rec文件（训练文件）

	4.制作property (num，inputsize): txt文件.

	5.增加保存多一份图片和对应关键点 （float32 to int64） 到新的文件夹用于DCNN训练（.pnf or .jpg 和 .pts） 




###人脸检测器:（4）
          SCRFD,path:/home/ye/Desktop/Model/insightface_new/detection/scrfd

	  或者 Video_track/coordinateReg/inference.py:之前写的视频跟踪GUI，里面有scrfd人脸检测框推理和坐标点转
          换过程，

          MTCNN,path:/home/ye/Desktop/Model/SDU/heatmapReg/mtcnn-model,里面有MTCNN人脸检测模型，和使用了
	             mtcnn人脸检测推理  


	1.使用两种人脸检测器制作固定人脸框，并计算失败率和推理速度
  		输入：文件夹内图片为总样本
  		输出：计算人脸检测器的失败率（失败样本/总样本，当阈值小于0.5为检测失败样本）
      		     单张图片推理时间

	2.可视化图像，
	输入：单张图片
	输出：在单张图片上显示SCRDF 和 MTCNN 中多个候选框的位置（蓝色），和最后确定的人脸检测框（红色）（可参考
        scrfd.py 中 softmax和nms（输入预测框，排序找出与真实框最接近的预测框））

	3.修改模型测试时人脸检测器为SCRFD


###人脸关键点检测模型：（7）
	           DAN_V2,path：/home/ye/Desktop/Model
		   heatmapReg,path：/home/ye/Desktop/Model/SDU

        1.可视化DAN中间层输出，输入：批量图片可以输出三种图 (是stage1的输出，也是当作stage2的输入)
		      输出：1.是 s2_lmark（经过affine 变换的landmarks和shape）
		            2.是 s2_heatmap（热力图：是landmark在人脸的分布情况）
		            3.是 s2_feature （第一个全链接层的输出）

        2.可视化SDU,输入：图片和模型文件：
                     输出：可视化指定网络层效果，（可以 打印网络层查询）
 
###DAN模型训练：
	Preprocess：
		python preprocessing.py --input_dir=... --output_dir=... --istrain=True --repeat=10 --img_size=112 --mirror_file=./Mirror68.txt
	Train model：
		python DAN_V2.py -ds 1 --data_dir=preprocess_output_dir --data_dir_test=...orNone -nlm 68 -te=15 -epe=1 -mode train
		python DAN_V2.py -ds 2 --data_dir=preprocess_output_dir --data_dir_test=...orNone -nlm 68 -te=45 -epe=1 -mode train




###评估测试集(图片包括.jpg 或.png)：
(三种模型，DCNN，DAN，SDU，三种数据集，每种分成10个测试集)   ### 10个测试集
       
        	0.DAN 和 heatmapReg 模型的批量数据预测 ：
        	原代码:DAN_V2/DAN_V3_img.py：原DAN模型单张测试代码，单张图片输入测试 改为批量输入图片 ###增加人脸检测器
        	heatmaoReg/test_batch_v2.py：原SDU批量测试代码，批量图片输入测试 ####替换人脸检测器
           
        	输入：一个文件夹内的测试集，可用./data文件夹，和./menpo_semi...zip ###输入仅支持图片文件
         
        	人脸检测模型为scrfd    在SCRFD.ZIP
        	人脸关键点模型可切换为dan(DAN_V2 模型文件)或者A-0001，A-0150（SDU模型文件）
       
        	输出：保存该文件夹下图片和预测点， （保存十个测试集的图片和文件）### 保存图片和pts
	


		
        1.制作误差值表格（2）：
		原代码:batchError_test.py 输入单个文件夹内图片和预测pts文件，计算三种误差值，保存表格
		      Organ_fig.py 输入单个文件内图片和预测pts文件计算每个器官的三种误差值，保存表格

		输入：该数据集的十个测试集的真实点和预测点，（二十个文件夹，每二个文件夹内包含一个测试集的图片和真实点，另一个包含图片和预测点）

		对每张图片进行误差计算，得出每张图的RMES，NME_ION, NME_IPN（每张图片68个点的三种误差值，保存在第一种表格）,

		然后计算出每个测试集的平均 RMSE,NME_ION, NME_IPN （平均三种误差值，保存在第二种表格）,
		然后计算出该十个测试集的平均 RMSE,NME_ION, NME_IPN （十个测试集的平均三种误差值，保存在第二种表格）.

		对十个测试集中的器官上的点进行误差计算，得出每张图中每个器官的平均RMES值 （每个器官上点的平均RMSE值，保存在第三种表格）


		输出三种表格: 1.十个测试集的每张图片68个点的三种误差值 （用于箱型图）
            	   	    2.每个测试集的平均三种误差值，十个测试集的平均三种误差。
            	    	    3.十个测试集的每个器官上点的平均RMSE值 （用于柱状图）
					（重复三次，得出三种模型的十个测试集，共九个测试集的误差表格）


        2.画箱型图和柱状图（3）
		输入：三种方法的十个测试集的每张图片68个点的三种误差值的表格：


		一个图中有每个方法的一种误差值的箱子，一共三种颜色的箱子

		箱型图：（five number 五个范围数字，范围为误差的最大值和最小值，箱子的范围为三个中位数，
		(中间的中位数是中间两个数的均值，左边的中位数是以中间中位数为界限，左边的误差值的中位数，右边的中位数是以中间中位数为界限，右边的误差值的中位数)

		the median is the mean of the middle two numbers. 四分数（quartile）is the median of the da
        -ta points to the left/right of the median.）

		输出箱型图：（一种数据集有三张箱型图：分别是 
					    1.三种方法在十个测试集上的68个点的平均RMSE值的箱型图，（每种方法使用同一种符号）
			  	            2.三种方法在十个测试集上的68个点的平均NME_ION值的箱型图,
			  	            3.三种方法在十个测试集上的68个点的平均NME_IPN值的箱型图.
 			  	            重复3种数据集，共9张图）



		输入：三种方法的十个测试集的每个器官上点的平均RMSE值的表格

		一个图中有三种颜色的柱子，每种颜色代表一种方法。每种柱子代表一个器官上的点。以点数为数量，值为每个点上的平均RMSE值。


		输出柱状图：(一种数据集有六张柱状图: 分别是:
					   1.三种方法在该十个测试集上测试68个点的平均RMSE值，y-axis为值，x-axis为点数（1-68），
					  （每个点为柱子，每种方法使用同一颜色,三种方法堆叠在一个柱子上）

					   2.三种方法在该十个测试集图片中轮廓上点的平均RMSE值，y-axis为值，x-axis为点数（1-17），同上

					   3.三种方法在该十个测试集图片中眉毛上点的平均RMSE值，y-axis为值，x-axis为点数（18-27），同上

					   4.三种方法在该十个测试集图片中眼睛上点的平均RMSE值，y-axis为值，x-axis为点数（28-36），同上

					   5.三种方法在该十个测试集图片中测试鼻子上点的平均RMSE值，y-axis为值，x-axis为点数（37-48），同上

					   6.三种方法在该十个测试集图片中嘴巴上点的平均RMSE值，y-axis为值，x-axis为点数（49-68），同上

					   重复3种数据集，共18张图）
					   
###Benchmark vs the optimal method:

	在相同的数据集训练的情况下，用MulitPLE和COFW，对比OpenFace和SDU，68个点（CED）和各个器官（柱状图


###Demo:

	inference.py 显示和保存出68个点的坐标位置 (68点的位置的顺序)
			  
	68 points ：0,2,4,6,8,9,11,13,15,18,20,22,24,25,27,29,31,35,(40+41)/2,(40+42)/2,39,(33+37)/2,(33+36)/2,43,(48+44)/2,(45+49)/2,(47+51)/2
	,(46+50)/2), 52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,78,79,80,84,85,86,87,89,(94+95)/2,(96+94)/2,93, (87+90)/2,
	(87+91)/2,(102+97)/2,(103+98)/2,(104+99)/2,(105+100)/2,101,

