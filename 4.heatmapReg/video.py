import argparse
import cv2
import sys
import numpy as np
import os
import mxnet as mx
import datetime
import img_helper
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'deploy'))
from mtcnn_detector import MtcnnDetector
import time

class Handler:
  def __init__(self, prefix, epoch, ctx_id=0):
    print('loading',prefix, epoch)
    if ctx_id>=0:
      ctx = mx.gpu(ctx_id)
    else:
      ctx = mx.cpu()
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers['heatmap_output']
    image_size = (128, 128)
    self.image_size = image_size
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    #model = mx.mod.Module(symbol=sym, context=ctx)
    model.bind(for_training=False, data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    self.model = model
    mtcnn_path = os.path.join(os.path.dirname(__file__), '.', 'mtcnn-model')
    self.det_threshold = [0.1,0.1,0]#threshold
    self.detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=self.det_threshold)#检测人脸框
  
  def get(self, img):
    ret = self.detector.detect_face(img, det_type = 0)
    # print(ret)
    if ret is None:
      return(None, None)
    bbox, points = ret
    if bbox.shape[0]==0:
      return None
    bbox = bbox[0,0:4]                             #Reszie the facial bouding box
    points = points[0,:].reshape((2,5)).T
    M = img_helper.estimate_trans_bbox(bbox, self.image_size[0], s = 2.0)
    rimg = cv2.warpAffine(img, M, self.image_size, borderValue = 0.0)
    img = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2,0,1)) #3*112*112, RGB
    input_blob = np.zeros( (1, 3, self.image_size[1], self.image_size[0]),dtype=np.uint8 )
    input_blob[0] = img
    ta = datetime.datetime.now()
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    alabel = self.model.get_outputs()[-1].asnumpy()[0]
    tb = datetime.datetime.now()
    print('module time cost', (tb-ta).total_seconds())
    ret = np.zeros( (alabel.shape[0], 2), dtype=np.float32)
    for i in range(alabel.shape[0]):
      a = cv2.resize(alabel[i], (self.image_size[1], self.image_size[0]))
      ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
      #ret[i] = (ind[0], ind[1]) #h, w
      ret[i] = (ind[1], ind[0]) #w, h
    return ret, M, bbox



class video_pre(object):
    def __init__(self):
        self.ctx_id = 0
        self.handler = Handler('./model/A', 150, self.ctx_id)
        self.result_img = "./video-result"

    def predictor(self, img, depth_img):
        try:
            imgname = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
            
            # print(imgname)
            result = []
            result.append('version: 1\n')                       #'version: 1\n');
            result.append('n_points:  68\n')                    #'n_points:  68\n');
            result.append('{\n')                                #'{\n');
            landmark, M, bbox = self.handler.get(img)
            # print(landmark, M)
            IM = cv2.invertAffineTransform(M)
            for i in range(landmark.shape[0]):
                p = landmark[i]
                point = np.ones( (3,), dtype=np.float32)
                point[0:2] = p
                point = np.dot(IM, point)
                landmark[i] = point[0:2]

            for i in range(landmark.shape[0]):
                p = landmark[i]
                point = (int(p[0]), int(p[1])) #save prediction points int type
                # print(img.shape)
                result.append('%.2f %.2f\n' % (point[0], point[1]))
                cv2.circle(img, point, 1, (0, 255, 0), 1)       #channel  3
                cv2.circle(depth_img, point, 1, (0, 255, 0), 1) #channel  1
            cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0,255,0), 2) #bbox[0:3] x,y,w,h 画框
            cv2.rectangle(depth_img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0,255,0), 2)
            cv2.imwrite("{}/{}.png".format(self.result_img, imgname), img)
            cv2.imwrite("{}/{}_depth.png".format(self.result_img, imgname), depth_img)
            with open("{}/{}.pts".format(self.result_img, imgname), "w") as f:
              for r in result:
                f.write(r)
        except:
            img = img
            depth_img = depth_img

        return img, depth_img

    def devio_run(self):
        cap = cv2.VideoCapture(1)
        while True:
            ret, frame = cap.read()
            #img = self.predictor(frame)
            cv2.imshow("cap", frame)
            if cv2.waitKey(100) & 0xff == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    p = video_pre()
    p.devio_run()


