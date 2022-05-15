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
import xlwt


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
    self.det_threshold = [0.1,0.1,0]
    self.detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=self.det_threshold)
  
  def get(self, img):
    ret = self.detector.detect_face(img, det_type = 0)
    print(ret)
    if ret is None:
      return(None, None)
    bbox, points = ret
    if bbox.shape[0]==0:
      return None
    bbox = bbox[0,0:4]                                 ##Reszie the facial bouding box
    #x,y,w,h = bbox[0:]
    #bbox = [x-10,y-10,w+20,h+20]   [int(0.9x),int(0.9y),int(1.1w),int(1.1h)]
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

    #title
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

ctx_id = 0
#path
img_path = 'test-single/001_02_02_051_19.jpg'
img = cv2.imread(img_path)
#img = np.zeros( (128,128,3), dtype=np.uint8 )
#print(img.shape)
handler = Handler('./model/A', 150, ctx_id) #loading model
for _ in range(10):
  ta = datetime.datetime.now()
  landmark, M, bbox = handler.get(img)
  tb = datetime.datetime.now()
  print('get time cost', (tb-ta).total_seconds())
#visualize landmark
try:
  result = []
  # visualize groundtruth
  label_path = "{}.pts".format(img_path.split(".")[0])
  true_points = open(label_path, "r").readlines()
  true_points = true_points[3:-1]

  result.append('version: 1\n')                       #'version: 1\n');
  result.append('n_points:  68\n')                                       #'n_points:  68\n');
  result.append('{\n')    
  IM = cv2.invertAffineTransform(M)
  for i in range(landmark.shape[0]):
    p = landmark[i]
    point = np.ones((3,), dtype=np.float32)
    point[0:2] = p
    point = np.dot(IM, point)
    landmark[i] = point[0:2]
  information_result = []
  for i in range(landmark.shape[0]):
    true_re = "{:.3f},{:.3f}".format(float(true_points[i].strip().split()[0]), float(true_points[i].strip().split()[1]))
    xAxis = float(true_points[i].split()[0]) - landmark[i][0]
    yAxis = float(true_points[i].split()[1]) - landmark[i][1]
    #Groundtruth', u'Prediction', u'Error Distance'
    information_result.append([i, true_re, str("{:.3f}".format(landmark[i][0])) + "," + str(
      "{:.3f}".format(landmark[i][1])), str("{:.3f}".format(np.sqrt(xAxis ** 2 +yAxis ** 2)))])
    p = landmark[i]
    point = (int(p[0]), int(p[1]))
    cv2.circle(img, point, 1, (0, 0, 255), -1)# prediction red
    cv2.circle(img, (int(float(true_points[i].strip().split()[0])), int(float(true_points[i].strip().split()[1]))), 1,(255, 0, 0), -1)# groundtruth blue
    result.append('%.2f %.2f\n' % (point[0], point[1]))
    #cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0,255,0), 2)#show bbox

except:
  pass
filename = "{}/{}".format("test-single-result", img_path.split("/")[-1]) #img path = test-single-result/*****.png
print('writing', filename)
cv2.imwrite(filename, img) #save img to test-single-result
write_excel('{}_excel'.format(filename.split(".")[0]), information_result) #write & save img from test-single-result
with open("{}.pts".format(img_path.split(".")[0]), "w") as f: #open test-single/*****.pts
              for r in result:
                f.write(r)



