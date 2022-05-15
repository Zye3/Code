import os
from mtcnn_detector import MtcnnDetector
import time
import mxnet as mx
import cv2
import insightface
import numpy as np

class model_evaluate(object):
    def __init__(self, ctx_id=0):
        if ctx_id >= 0:
            ctx = mx.gpu(ctx_id)
        else:
            ctx = mx.cpu()
        # load mtcnn model
        mtcnn_path = os.path.join(os.path.dirname(__file__), '.', 'mtcnn-model')
        self.det_threshold = [0.1, 0.1, 0.5]  # threshold
        self.mtcnn_detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=self.det_threshold)#检测人脸框
        # load scrfd model
        self.det_size = 224
        scrfd_model = r'./scrfd-model/scrfd_500m_shape640x640.onnx'
        self.scrfd_model = insightface.model_zoo.get_model(scrfd_model)
        self.scrfd_model.prepare(ctx_id=ctx_id)

    def mtcnn_predict(self, img):
        bboxes, points = self.mtcnn_detector.detect_face(img, det_type=0)
        return bboxes

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

    def compute_iou(self, rec1, rec2):
        """
        computing IoU
        :param rec1: (y0, x0, y1, x1), which reflects
                (top, left, bottom, right)
        :param rec2: (y0, x0, y1, x1)
        :return: scala value of IoU
        """
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

        # computing the sum_area
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])

        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return (intersect / (sum_area - intersect)) * 1.0

    def task_one(self, img_path):
        """
        eval model
        """
        # save path
        save_path = "test_result"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        if os.path.isdir(img_path):
            files = ["{}/{}".format(img_path, i) for i in os.listdir(img_path) if i.split(".")[-1] in ["jpg", "png"]]
        else:
            files = [img_path]
        mtcnn_t = []
        scrfd_t = []
        mtcnn_iou_list = []
        scrfd_iou_list = []
        for file in files:
            print(file)
            img = cv2.imread(file)
            # load box
            label_file = "{}/{}.pts".format(img_path, file.split("/")[-1].split(".")[0])
            with open(label_file, "r") as f:
                data = f.readlines()
            points = data[3:-1]
            x_min, x_max, y_min, y_max = 1000000, 0, 1000000, 0
            for m in points:
                m = m.strip().split(" ")
                if float(m[0]) < x_min:
                    x_min = float(m[0])
                if float(m[0]) > x_max:
                    x_max = float(m[0])
                if float(m[1]) < y_min:
                    y_min = float(m[1])
                if float(m[1]) > y_max:
                    y_max = float(m[1])
            box = [x_min, y_min, x_max, y_max]
            """
            inference mtcnn
            """
            t1 = time.time()
            mtcnn_bboxes = self.mtcnn_predict(img)
            t2 = time.time()
            mtcnn_t.append(t2 - t1)
            """
            inference scrfd
            """
            det_im, det_scale = self.square_crop(img, self.det_size)
            scrfd_bboxes, _ = self.scrfd_model.detect(det_im)
            scrfd_bboxes /= det_scale
            t3 = time.time()
            scrfd_t.append(t3 - t2)
            boxes = [box]
            for box in boxes:
                mt_box = []
                mt_iou = 0
                sc_box = []
                sc_iou = 0
                for pred in mtcnn_bboxes:
                    mtcnn_box = pred[0:4]
                    mtcnn_iou = self.compute_iou(mtcnn_box, box)
                    if mtcnn_iou > 0.0 and mtcnn_iou > mt_iou:
                        mt_box = mtcnn_box
                        mt_iou = mtcnn_iou
                mtcnn_iou_list.append(mt_iou)
                for pred in scrfd_bboxes:
                    scrfd_box = pred[0:4]
                    scrfd_iou = self.compute_iou(scrfd_box, box)
                    if scrfd_iou > 0.0 and scrfd_iou > sc_iou:
                        sc_box = scrfd_box
                        sc_iou = scrfd_iou
                scrfd_iou_list.append(sc_iou)
                # color bule
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0))
                cv2.putText(img, 'label', (int(box[0]), int(box[1]+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                # color gred
                if mt_box != []:
                    cv2.rectangle(img, (int(mt_box[0]), int(mt_box[1])), (int(mt_box[2]), int(mt_box[3])), (0, 255, 0))
                    cv2.putText(img, 'mtcnn', (int(mt_box[0]), int(mt_box[1]+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # color green
                if sc_box != []:
                    cv2.rectangle(img, (int(sc_box[0]), int(sc_box[1])), (int(sc_box[2]), int(sc_box[3])), (0, 0, 255))
                    cv2.putText(img, 'scrfd', (int(sc_box[0]), int(sc_box[1]+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            save_file = "{}/{}".format(save_path, file.split("/")[-1])
            cv2.imwrite(save_file, img)


        print("mtcnn model cost time: {:.5f}, miou: {:.5f}; scrfd model cost time: {:.5f}, miou: {:.5f}" \
              .format(np.mean(mtcnn_t), np.mean(scrfd_t), np.mean(mtcnn_iou_list), np.mean(scrfd_iou_list)))


if __name__ == "__main__":
    model = model_evaluate()
    path = "test"
    model.task_one(path)