import dan_model
import dan_run_loop_modified

import os
import sys
import glob

import numpy as np
import cv2
import tensorflow as tf
# 载入人脸分类器
import dlib



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


def main(argv):
    parser = dan_run_loop_modified.DANArgParser()
    parser.set_defaults(data_dir='./data_dir',
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
    modelnet = vgg16_model_fn
    # 获取人脸检测器
    detector = dlib.get_frontal_face_detector()
    if flags.mode == tf.estimator.ModeKeys.PREDICT:
        # 调用摄像头
        video = cv2.VideoCapture(0)
        while video.isOpened():
            ret, img = video.read()
            cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 检测人脸
            dets = detector(cv_img, 1)

            # 将每一个人脸标注
            for face in dets:
                left = face.left()
                top = face.top()
                right = face.right()
                bottom = face.bottom()
                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                # 将人脸部分图像切割送给模型检测关键点
                crop_img = cv_img[top:bottom, left:right]
                input_function = video_input_fn(crop_img, 112, flags.num_lmark)
                predict_results = dan_run_loop_modified.dan_main(flags, modelnet, input_function)
                # 计算图像和送进模型的图像比例
                h_proportion, w_proportion = crop_img.shape[0] / 112.0, crop_img.shape[1] / 112.0
                # 将检测到的关键点画在原图像上
                for x in predict_results:
                    landmark = x['s2_ret']
                    for lm in landmark:
                        cv2.circle(img, (int(lm[0]*h_proportion+left), int(lm[1]*w_proportion+top)), 1, (0,0,255), -1)
            # 显示
            cv2.imshow('Video', img)
            # 200ms一张图像，这里可调，去觉得计算机计算速度，跟不上就会卡顿，延时，按q按键退出
            if cv2.waitKey(600) & 0xFF == ord('q'):
                break

        video.release()


    # dan_run_loop_modified.dan_main(flags,vgg16_model_fn,input_function)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(argv=sys.argv)
