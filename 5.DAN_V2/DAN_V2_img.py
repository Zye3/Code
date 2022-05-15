import dan_model
import dan_run_loop_modified

import os
import sys
import glob
# 载入人脸分类器
import dlib
import numpy as np
import cv2
import tensorflow as tf
import xlwt
import pandas as pd


class VGG16Model(dan_model.Model):
    def __init__(self,num_lmark,data_format=None):
        
        img_size=112
        filter_sizes=[64,128,256,512]
        num_convs=2
        kernel_size=3

        super(VGG16Model,self).__init__(
            num_lmark=num_lmark,
            img_size=img_size,
            filter_sizes=filter_sizes,
            num_convs=num_convs,
            kernel_size=kernel_size,
            data_format=data_format
        )

def get_filenames(data_dir):
    listext = ['*.png','*.jpg']

    imagelist = []
    for ext in listext:
        p = os.path.join(data_dir, ext)
        imagelist.extend(glob.glob(p))

    ptslist = []
    for image in imagelist:
        ptslist.append(os.path.splitext(image)[0] + ".ptv")

    return imagelist, ptslist


def get_synth_input_fn():
    return dan_run_loop_modified.get_synth_input_fn(112, 112, 1, 68)

def vgg16_input_fn(is_training,data_dir,batch_size=64,num_epochs=1,num_parallel_calls=1, multi_gpu=False):
    img_path,pts_path = get_filenames(data_dir)

    def decode_img_pts(img,pts,is_training):
        img = cv2.imread(img.decode(), cv2.IMREAD_GRAYSCALE)
        pts = np.loadtxt(pts.decode(),dtype=np.float32,delimiter=',')
        return img[:,:,np.newaxis].astype(np.float32),pts.astype(np.float32)

    map_func=lambda img,pts,is_training:tuple(tf.py_func(decode_img_pts,[img,pts,is_training],[tf.float32,tf.float32]))

    img = tf.data.Dataset.from_tensor_slices(img_path)
    pts = tf.data.Dataset.from_tensor_slices(pts_path)

    dataset = tf.data.Dataset.zip((img, pts))
    num_images = len(img_path)

    return dan_run_loop_modified.process_record_dataset(dataset,is_training,batch_size,
                                               num_images,map_func,num_epochs,num_parallel_calls,
                                               examples_per_epoch=num_images, multi_gpu=multi_gpu)

def read_dataset_info(data_dir):
    mean_shape = np.loadtxt(os.path.join(data_dir,'mean_shape.ptv'),dtype=np.float32,delimiter=',')
    imgs_mean = np.loadtxt(os.path.join(data_dir,'imgs_mean.ptv'),dtype=np.float32,delimiter=',')
    imgs_std = np.loadtxt(os.path.join(data_dir,'imgs_std.ptv'),dtype=np.float32,delimiter=',')
    return mean_shape.astype(np.float32) ,imgs_mean.astype(np.float32),imgs_std.astype(np.float32)


def video_input_fn(img, img_size, num_lmark):

    def _get_frame():
        frame = cv2.resize(img, (img_size, img_size)).astype(np.float32)
        yield (frame, np.zeros([num_lmark, 2], np.float32))

    def input_fn():
        dataset = tf.data.Dataset.from_generator(_get_frame,(tf.float32,tf.float32),(tf.TensorShape([img_size,img_size]),tf.TensorShape([num_lmark,2])))
        return dataset

    return input_fn

# 设置excel表格宽度自适应标题行
def len_byte(value):
    length = len(value)
    utf8_length = len(value.encode('utf-8'))
    length = (utf8_length - length) / 2 + length
    return (int(length))


# 设置表格格式
def set_style(name, height, bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.height = height
    alignment = xlwt.Alignment()
    alignment.horz = xlwt.Alignment.HORZ_CENTER
    style.font = font
    style.alignment = alignment
    return (style)


# excel写操作
def write_excel(file_path, video_information):
    excel_file = xlwt.Workbook()
    sheet1 = excel_file.add_sheet(u'sheet1', cell_overwrite_ok=True)
    # title
    row0 = [u'Filename', u'Groundtruth', u'Prediction', u'Error Distance']
    col_width = []
    for i in range(len(row0)):
        col_width.append(len_byte(row0[i]))
    for i in range(len(col_width)):
        if col_width[i] > 5:
            sheet1.col(i).width = 256 * (col_width[i] + 5)
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))

    for i in range(len(video_information)):
        video_maths = video_information[i]
        i = i + 1
        for j in range(len(video_maths)):
            video_math = video_maths[j]
            sheet1.write(i, j, video_math)

    file_name_path = file_path + '.xls'
    excel_file.save(file_name_path)
    return (row0)

def main(argv):
    parser = dan_run_loop_modified.DANArgParser()
    parser.set_defaults(data_dir='/data_dir',#'./test_processed_uclan/uclan (4)@6ea2748a-f4be-11ea-b44c-bbbdaf08f92c.png',
                        model_dir='./model_dir',
                        data_format='channels_last',
                        train_epochs=20,
                        epochs_per_eval=10,
                        batch_size=64)

    flags = parser.parse_args(args=argv[1:])

    mean_shape = None
    imgs_mean = None
    imgs_std = None

    flags_trans = { 
        'train':tf.estimator.ModeKeys.TRAIN,
        'eval':tf.estimator.ModeKeys.EVAL,
        'predict':tf.estimator.ModeKeys.PREDICT
                  }

    flags.mode = flags_trans[flags.mode]

    # 获取人脸检测器
    # detector = dlib.get_frontal_face_detector()

    if flags.mode == tf.estimator.ModeKeys.TRAIN:
        mean_shape,imgs_mean,imgs_std = read_dataset_info(flags.data_dir)

    def vgg16_model_fn(features, labels, mode, params):
        return dan_run_loop_modified.dan_model_fn(features=features,
                            groundtruth=labels,
                            mode=mode,
                            stage=params['dan_stage'],                                                    
                            num_lmark=params['num_lmark'],
                            model_class=VGG16Model,
                            mean_shape=mean_shape,
                            imgs_mean=imgs_mean,
                            imgs_std=imgs_std,
                            data_format=params['data_format'],
                            multi_gpu=params['multi_gpu'])

    input_function = flags.use_synthetic_data and get_synth_input_fn() or vgg16_input_fn


    if flags.mode == tf.estimator.ModeKeys.PREDICT:

        print("*"*40)
        print(flags.data_dir)
        # 将人脸部分图像切割送给模型检测关键点
        img = cv2.imread(flags.data_dir)
        path_img = cv2.imread(flags.data_dir, 0)
        print(path_img)
        input_function = video_input_fn(path_img, 112, flags.num_lmark)
        predict_results = dan_run_loop_modified.dan_main(flags, vgg16_model_fn, input_function,
                                                         file_path=flags.data_dir)
        # 将检测到的关键点画在原图像上
        print(predict_results)
        for x in predict_results:
            landmark = x['s2_ret']
            # print(landmark)
            for lm in landmark:
                cv2.circle(img, (int(lm[0]), int(lm[1])), 1, (0,0,255), -1)
        #visual groundtruth
        label_path = flags.data_dir.replace(flags.data_dir[-3:], 'ptv')#loading gt's file.ptv
        # img = os.path.join(flags.data_dir, img)
        true_points = pd.read_csv(label_path, sep=',', header=None)#change to Axis
        true_points = true_points.values.astype(np.int64)#save points

        information_result = []
        #calculate
        for i in range(len(landmark)):
            # true_re = "{:.3f},{:.3f}".format(float(true_points[i].strip().split()[0]), float(true_points[i].strip().split()[1]))
            # xAxis = float(true_points[i].split()[0]) - (landmark[i][0])
            # yAxis = float(true_points[i].split()[1]) - (landmark[i][1])
            true_re = "{:.3f},{:.3f}".format(float(true_points[i][0]),#[i][0][1]=x,y of the i image
                                             float(true_points[i][1]))
            xAxis = float(true_points[i][0]) - (landmark[i][0])
            yAxis = float(true_points[i][1]) - (landmark[i][1])
            information_result.append([i, true_re, str("{:.3f}".format(landmark[i][0]))+","+str("{:.3f}".format(landmark[i][1])),
                                       str("{:.3f}".format(np.sqrt(xAxis ** 2 +yAxis ** 2)))])#Filename', u'Groundtruth', u'Prediction', u'Error Distance
            # cv2.circle(img, (int(float(true_points[i].strip().split()[0])), int(float(true_points[i].strip().split()[1]))), 1, (255, 0, 0), -1)
            cv2.circle(img, (int(float(true_points[i][0])), int(float(true_points[i][1]))), 1, (255, 0, 0), -1)#groundtruth blue
        filename = "{}/{}/{}".format("results","test_single_result",flags.data_dir.split("/")[-1])  # img path = test-single-result/*****.png
        cv2.imwrite(filename,img)
        write_excel('{}_excel'.format(filename.split(".")[0]),information_result)
        # cv2.imwrite( './results/test_single_result/'+"uclan(10)"+'_pred.png', img) #save img
        # write_excel('./results/test_single_result/'+"uclan(10)"+"_excel", information_result) #save excel
        # np.savetxt('./results/test_single_result/'+"uclan(10)"+'_pred.pts', landmark , delimiter=" ", fmt='%i') #save predicted .pts file



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(argv=sys.argv)
