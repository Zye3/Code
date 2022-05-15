import mxnet as mx
from mxnet import recordio
import numpy as np
import cv2

path_imgidx = "./train.idx"
path_imgrec = "./train.rec"

imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
oseq = list(imgrec.keys)
print('train size', len(oseq))

s = imgrec.read_idx(2)
header, img = recordio.unpack(s)
img = mx.image.imdecode(img).asnumpy()
hlabel = np.array(header.label).reshape((68, 2))
print(header, img)

for i in hlabel:
    print(i)
    cv2.circle(img, (i[0], i[1]), 2, (255, 255, 255))
cv2.imwrite("1.jpg", img)