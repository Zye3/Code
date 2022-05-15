import argparse
import cv2
import sys
import numpy as np
import os
import mxnet as mx
import datetime
import img_helper

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'deploy'))
# from mtcnn_detector import MtcnnDetector
import insightface

class Handler:
    def __init__(self, prefix, epoch, ctx_id=0):
        """

        """
        print('loading', prefix, epoch)
        if ctx_id >= 0:
            ctx = mx.gpu(ctx_id)
        else:
            ctx = mx.cpu()
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers['heatmap_output']
        image_size = (128, 128)
        self.image_size = image_size
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        # model = mx.mod.Module(symbol=sym, context=ctx)
        model.bind(for_training=False, data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model
        # mtcnn_path = os.path.join(os.path.dirname(__file__), '.', 'mtcnn-model')
        # self.det_threshold = [0.1, 0.1, 0]
        # self.detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True,
        #                               threshold=self.det_threshold)
        """
        20220508: scrfd model
        """
        self.det_size = 224
        scrfd_model = r'./scrfd-model/scrfd_500m_shape640x640.onnx'
        self.scrfd_model = insightface.model_zoo.get_model(scrfd_model)
        self.scrfd_model.prepare(ctx_id=ctx_id)

    def square_crop(self, im, S):
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

    def get(self, img):
        # ret = self.detector.detect_face(img, det_type=0)
        """
        20220508: scrfd predict
        """
        det_im, det_scale = self.square_crop(img, self.det_size)
        ret, _ = self.scrfd_model.detect(det_im)
        ret /= det_scale
        if ret is None:
            return None, None, None
        bbox = ret
        if bbox.shape[0] == 0:
            return None, None, None
        bbox = bbox[0, 0:4]  ##Reszie the facial bouding box
        # x,y,w,h = bbox[0:]
        # bbox = [x-10,y-10,w+20,h+20]   [int(0.9x),int(0.9y),int(1.1w),int(1.1h)]
        # points = points[0, :].reshape((2, 5)).T
        M = img_helper.estimate_trans_bbox(bbox, self.image_size[0], s=2.0)
        rimg = cv2.warpAffine(img, M, self.image_size, borderValue=0.0)
        img = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        input_blob = np.zeros((1, 3, self.image_size[1], self.image_size[0]), dtype=np.uint8)
        input_blob[0] = img
        ta = datetime.datetime.now()
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        alabel = self.model.get_outputs()[-1].asnumpy()[0]
        tb = datetime.datetime.now()
        # print('module time cost', (tb - ta).total_seconds())
        ret = np.zeros((alabel.shape[0], 2), dtype=np.float32)
        for i in range(alabel.shape[0]):
            a = cv2.resize(alabel[i], (self.image_size[1], self.image_size[0]))
            ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
            # ret[i] = (ind[0], ind[1]) #h, w
            ret[i] = (ind[1], ind[0])  # w, h
        return ret, M, bbox


if __name__ == "__main__":
    ctx_id = 0
    handler = Handler('./model/A', 150, ctx_id)
    path = './test' #loading path
    for file in os.listdir(path): #load image only
        img_path = "{}/{}".format(path, file)
        img = cv2.imread(img_path)
        for _ in range(10):
            ta = datetime.datetime.now()
            landmark, M, bbox = handler.get(img)
            tb = datetime.datetime.now()
            print('get time cost', (tb - ta).total_seconds())
        # visualize landmark
        # try:
        result = []
        result.append('version: 1\n')  # 'version: 1\n');
        result.append('n_points:  68\n')  # 'n_points:  68\n');
        result.append('{\n')
        # # visualize groundtruth
        # label_path = "{}.pts".format(img_path.split(".")[0])
        # true_points = open(label_path, "r").readlines()
        # true_points = true_points[3:-1]

        IM = cv2.invertAffineTransform(M)
        for i in range(landmark.shape[0]):
            p = landmark[i]
            point = np.ones((3,), dtype=np.float32)
            point[0:2] = p
            point = np.dot(IM, point)
            landmark[i] = point[0:2]

        for i in range(landmark.shape[0]):
            # true_re = "{:.3f},{:.3f}".format(float(true_points[i].strip().split()[0]),
            #                                  float(true_points[i].strip().split()[1]))
            p = landmark[i]
            point = (float(p[0]), float(p[1])) #int change to  float
            """
            opencv 画点必须是int型
            """
            cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 255, 0), 1) #prediction green
            # cv2.circle(img, (
            # int(float(true_points[i].strip().split()[0])), int(float(true_points[i].strip().split()[1]))), 1,
            #            (255, 0, 0), -1)  # groundtruth blue
            result.append('%.2f %.2f\n' % (point[0], point[1]))
        result.append('}')
        # cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

        # except:
        #     pass
        filename = "{}/{}_pred.png".format("test-result", file.split(".")[0])
        print('writing', filename)
        cv2.imwrite(filename, img)
        with open("{}/{}_pred.pts".format("test-result", file.split(".")[0]), "w") as f:
            for r in result:
                f.write(r)



