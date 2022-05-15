
import cv2
import numpy as np
from skimage import transform as trans

# def parse_lst_line(line):
#   vec = line.strip().split("\t")
#   assert len(vec)>=3
#   aligned = int(vec[0])
#   image_path = vec[1]
#   label = int(vec[2])
#   bbox = None
#   landmark = None
#   #print(vec)
#   if len(vec)>3:
#     bbox = np.zeros( (4,), dtype=np.int32)
#     for i in xrange(3,7):
#       bbox[i-3] = int(vec[i])
#     landmark = None
#     if len(vec)>7:
#       _l = []
#       for i in xrange(7,17):
#         _l.append(float(vec[i]))
#       landmark = np.array(_l).reshape( (2,5) ).T
#   #print(aligned)
#   return image_path, label, bbox, landmark, aligned
#



def read_image(img_path, **kwargs):
  mode = kwargs.get('mode', 'rgb')
  layout = kwargs.get('layout', 'HWC')
  if mode=='gray':
    img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  else:
    img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_COLOR)
    if mode=='rgb':
      #print('to rgb')
      img = img[...,::-1]
    if layout=='CHW':
      img = np.transpose(img, (2,0,1))
  return img


def preprocess(img, bbox=None, landmark=None, **kwargs):
  if isinstance(img, str):
    img = read_image(img, **kwargs)
  M = None
  image_size = []
  str_image_size = kwargs.get('image_size', '')
  if len(str_image_size)>0:
    image_size = [int(x) for x in str_image_size.split(',')]
    if len(image_size)==1:
      image_size = [image_size[0], image_size[0]]
    assert len(image_size)==2
    assert image_size[0]==112
    assert image_size[0]==112 or image_size[1]==96
  if landmark is not None:
    assert len(image_size)==2
    # src = np.array([
    #   [30.2946, 51.6963],
    #   [65.5318, 51.5014],
    #   [48.0252, 71.7366],
    #   [33.5493, 92.3655],
    #   [62.7299, 92.2041] ], dtype=np.float32 )
    src = np.float32([
      (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
      (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
      (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
      (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
      (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
      (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
      (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
      (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
      (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
      (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
      (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
      (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
      (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
      (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
      (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
      (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
      (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
      (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
      (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
      (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
      (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
      (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
      (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
      (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
      (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
      (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
      (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
      (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
      (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
      (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
      (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
      (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
      (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
      (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

    src *= 112
    if image_size[1]==112:
      src[:,0] += 8.0
    dst = landmark.astype(np.float32)

    # tform = trans.SimilarityTransform()
    # tform.estimate(dst, src)
    # M = tform.params[0:2,:]
    # M_1 = cv2.estimateRigidTransform(dst.reshape(1,68,2), src.reshape(1,68,2), False)

  if M is None:
    if bbox is None: #use center crop
      det = np.zeros(4, dtype=np.int32)
      det[0] = int(img.shape[1]*0.0625)
      det[1] = int(img.shape[0]*0.0625)
      det[2] = img.shape[1] - det[0]
      det[3] = img.shape[0] - det[1]
    else:
      det = bbox
    margin = kwargs.get('margin', 44)
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
    bb[3] = np.minimum(det[3]+margin/2, img.shape[0])

    ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
    """
    
    """
    dst[:, 0] -= bb[0]
    dst[:, 1] -= bb[1]
    w_th = image_size[1] / ret.shape[1] #w ratio
    h_th = image_size[0] / ret.shape[0] #h ratio
    dst[:, 0] *= w_th
    dst[:, 1] *= h_th
    if len(image_size)>0:
      ret = cv2.resize(ret, (image_size[1], image_size[0]))
    return ret, dst
  else: #do align using landmark
    assert len(image_size)==2

    #src = src[0:3,:]
    #dst = dst[0:3,:]


    #print(src.shape, dst.shape)
    #print(src)
    #print(dst)
    #print(M)
    # cv2.imwrite("1.jpg", img)
    warped = cv2.warpAffine(img, M, (image_size[1],image_size[0]), borderValue = 0.0)
    # cv2.imwrite("2.jpg", warped)

    #tform3 = trans.ProjectiveTransform()
    #tform3.estimate(src, dst)
    #warped = trans.warp(img, tform3, output_shape=_shape)
    return warped, M


