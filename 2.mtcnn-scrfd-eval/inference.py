#!/usr/bin/env python 

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'deploy'))
import time
import cv2
import numpy as np
import mxnet as mx
from skimage import transform as trans
import insightface
import datetime


def square_crop(im, S):
    if im.shape[0] > im.shape[1]:
        height = S
        width = int(float(im.shape[1]) / im.shape[0] * S)
        scale = float(S) / im.shape[0]
    else:
        width = S
        height = int(float(im.shape[0]) / im.shape[1] * S)
        scale = float(S) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((S, S, 3), dtype=np.uint8)
    det_im[:resized_im.shape[0], :resized_im.shape[1], :] = resized_im
    return det_im, scale


def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0)
    return cropped, M


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i] = new_pt[0:2]
    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale
    return new_pts

def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)


class Handler:
    def __init__(self, prefix, epoch, im_size=192, det_size=224, ctx_id=0, detector_model=r'./model/scrfd_500m_shape640x640.onnx'):
        print('loading detector model:{}, epoch:{};'.format(prefix, epoch))
        if ctx_id >= 0:
            ctx = mx.gpu(ctx_id)
        else:
            ctx = mx.cpu()
        image_size = (im_size, im_size)
        self.detector = insightface.model_zoo.get_model(detector_model)  # can replace with your own face detector
        self.detector.prepare(ctx_id=ctx_id)
        self.det_size = det_size
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        self.image_size = image_size
        self.model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        self.model.bind(for_training=False, data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        self.model.set_params(arg_params, aux_params)


    def get(self, img, get_all=False):
        out = []
        pred = []
        bbox = []
        det_im, det_scale = square_crop(img, self.det_size)
        # a = time.time()
        bboxes, _ = self.detector.detect(det_im)
        # print("Time to detect facial bbox is {}".format(time.time()-a))
        if bboxes.shape[0] == 0:
            return pred, bbox
        bboxes /= det_scale
        if not get_all:
            areas = []
            for i in range(bboxes.shape[0]):
                x = bboxes[i]
                area = (x[2] - x[0]) * (x[3] - x[1])
                areas.append(area)
            m = np.argsort(areas)[-1]
            bboxes = bboxes[m:m + 1]
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            input_blob = np.zeros((1, 3) + self.image_size, dtype=np.float32)
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = self.image_size[0] * 2 / 3.0 / max(w, h)
            rimg, M = transform(img, center, self.image_size[0], _scale, rotate)
            rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
            rimg = np.transpose(rimg, (2, 0, 1))  # 3*112*112, RGB
            input_blob[0] = rimg
            data = mx.nd.array(input_blob)
            db = mx.io.DataBatch(data=(data,))
            # a = time.time()
            self.model.forward(db, is_train=False)
            pred = self.model.get_outputs()[-1].asnumpy()[0]
            # print("Time to detect facial landamrk is {}".format(time.time()-a))
            if pred.shape[0] >= 3000:
                pred = pred.reshape((-1, 3))
            else:
                pred = pred.reshape((-1, 2))
            pred[:, 0:2] += 1
            pred[:, 0:2] *= (self.image_size[0] // 2)
            if pred.shape[1] == 3:
                pred[:, 2] *= (self.image_size[0] // 2)
            IM = cv2.invertAffineTransform(M)
            pred = trans_points(pred, IM)
            out.append(pred)
        return pred, bbox    #single pred bbox ,mutil-face out bboxs


class video_pre(object):
    def __init__(self):
        self.handler = Handler('./scrfd-model/2d106det', 0, ctx_id=0, det_size=640)

    def data_now(self):
        # get now time for img name
        date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        return date_time

    def predictor(self, rgb_img=None, depth_img=None, depth=False):
        """
        预测人脸框和特征点
        :param rgb_img: 通道为rgb的彩色图像
        :param depth_img: 单通道的灰度图像
        :return: rgb_img, depth_img, result     rgb_img: 标注人脸框和特征点的彩色图像, depth_img: 标注人脸框和特征点的深度图像, result: 特征点的坐标，保存为json格式;
        """
        result = []
        result.append('version: 1\n')                       #'version: 1\n');
        result.append('n_points:  68\n')                    #'n_points:  68\n');
        result.append('{\n')                                #'{\n');
        landmark, bbox = self.handler.get(rgb_img, get_all=True)
        if landmark == []:
            return rgb_img, depth_img, result
        for i in range(landmark.shape[0]):
            p = landmark[i]
            result.append('%.2f %.2f\n' % (int(p[0]), int(p[1])))
        #     # 在图上画特征点
        #     cv2.circle(rgb_img, (int(p[0]), int(p[1])), 1, (0, 0, 255), 2)       #channel  3
        #     if depth:
        #         cv2.circle(depth_img, (int(p[0]), int(p[1])), 1, (0, 0,255), 2)    #channel  1
        # # 在图上画人脸检测框
        # cv2.rectangle(rgb_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)   #bbox[0:3] x,y,w,h
        # if depth:
        #     cv2.rectangle(depth_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

        result.append('}')
        return rgb_img, depth_img, result

if __name__ == "__main__":
    p = video_pre()




