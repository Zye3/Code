# coding:utf-8
import os
import xlwt
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps
import os
import pandas as pd
import json
gt_dir = 'results/image'
res_dir = 'results/test-result-xm2vts'

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
def write_excel(file_path, video_information, row0=None):
    """

    :param file_path:
    :param video_information: all information, must be a list
    :param row0: tittle name
    :return:
    """
    excel_file = xlwt.Workbook()
    sheet1 = excel_file.add_sheet(u'sheet1', cell_overwrite_ok=True)

    # row0 = [u'源文件名', u'文件地址', u'帧率', u'分辨率', u'轨道', u'时间线帧率', u'时间线入点帧',
    #         u'时间线入点时码', u'时间线出点帧', u'时间线出点时码', u'源文件起始时码', u'源文件入点帧',
    #         u'源文件入点时码', u'源文件出点帧', u'源文件出点时码', u'时间线时长', u'片段帧数', u'总时长']
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

#
#DCNN Openface SDU

def get_Landmarks(img, gt_res):
    if gt_res == 'gt':
        img = img.replace(img[-4:], '.pts')
        img = os.path.join(gt_dir, img)
        with open(img, "r") as f:
             content = f.readlines()
        data = []

        # json pts file
        for i in content[3:-1]:#75points [10:-1] Mutilpie
            ''' 68 points,contour [3:20],brow [21:30],nose [31:39],eye [40:49], mouth [50:71]
                75 points,contour [10:27],brow [28:37],nose [38:46],eye [47:56], mouth [57:78]
            '''
            data.append([float(i.strip().split(" ")[0]), float(i.strip().split(" ")[1])])
        data = np.array(data, dtype=np.int64)
        return data


    elif gt_res == 'res':
        img = img.replace(img[-4:],'_pred.pts')
        # img = img.replace(img[-4:], '.pts') #DCNN
        img = os.path.join(res_dir, img)

        with open(img, "r") as f:
             content = f.readlines()
        data = []

        # json pts file
        for i in content[3:-1]:
            ''' 68 points,contour [3:20],brow [21:30],nose [31:39],eye [40:49], mouth [50:71]
            '''
            data.append([float(i.strip().split(" ")[0]), float(i.strip().split(" ")[1])])
        data = np.array(data, dtype=np.int64)
        return data


# # DAN
# def get_Landmarks(img, gt_res):
#     if gt_res == 'gt':
#         img = img.replace(img[-3:],'ptv')#replace filename from image to ptv
#         img = os.path.join(gt_dir,img) #open ptv file
#         data = pd.read_csv(img,sep=',',header=None)# format of reloading ptv file
#         data =data[7:]        #75 point
#         return data.values.astype(np.int64)
#     elif gt_res == 'res':
#         img = img.replace(img[-4:],'_pred.pts')
#         img = os.path.join(res_dir,img)
#
#         with open(img, "r") as f:
#             content = f.readlines()
#         data = []
#         for i in content[3:-1]:
#             data.append([float(i.strip().split(" ")[0]), float(i.strip().split(" ")[1])])
#         data = np.array(data, dtype=np.int64)
#         # with open(img, "r") as f:
#         #      content = f.readlines()
#         # data = []
#         # # No format pts file
#         # for i in content:
#         #     data.append([float(i.strip().split(" ")[0]), float(i.strip().split(" ")[1])])
#         # data = np.array(data, dtype=np.int64)
#         # data = pd.read_csv(img,sep=' ',header=None)
#         return data

def LandmarkError(img_list,normalization=None):
    """
    :param img_list:
    :param normalization: must be a list
    :param showResults:
    :param verbose:
    :return:
    """
    rsme_mean = []
    rsme_coutour_mean = []
    rsme_brow_mean = []
    rsme_nose_mean = []
    rsme_eye_mean = []
    rsme_mouth_mean = []

    errors_centers = []
    errors_corners = []
    errors_diagonal = []
    # errors_corners_coutour = []
    # errors_corners_brow = []
    # errors_corners_nose = []
    # errors_corners_eye = []
    # errors_corners_mouth = []
    errors = []

    # title
    one_tittle = ['imgname', 'RSME',        'NME_corner',        'NME_center',        'NME_diagonal',
                             'RSME_coutour','NME_corner_coutour','NME_center_coutour','NME_diagonal_coutour',
                             'RSME_brow',   'NME_corner_brow',   'NME_center_brow',   'NME_diagonal_brow',
                             'RSME_nose',   'NME_corner_nose',   'NME_center_nose',   'NME_diagonal_nose',
                             'RSME_eye',    'NME_corner_eye',    'NME_center_eye',    'NME_diagonal_eye',
                             'RSME_mouth',  'NME_corner_mouth',  'NME_center_mouth',  'NME_diagonal_mouth']
    one_information = []
    two_tittle = ['dirname', 'RSME',        'NME_corner',        'NME_center',        'NME_diagonal','AUC@(0.35)','Failure rate'
                             'RSME_coutour','NME_corner_coutour','NME_center_coutour','NME_diagonal_coutour',
                             'RSME_brow',   'NME_corner_brow',   'NME_center_brow',   'NME_diagonal_brow',
                             'RSME_nose',   'NME_corner_nose',   'NME_center_nose',   'NME_diagonal_nose',
                             'RSME_eye',    'NME_corner_eye',    'NME_center_eye',    'NME_diagonal_eye',
                             'RSME_mouth',  'NME_corner_mouth',  'NME_center_mouth',  'NME_diagonal_mouth']
    two_information = []

    for i, img in enumerate(img_list):
        information = []
        information.append(img)
        gtLandmarks = get_Landmarks(img, 'gt')
        resLandmarks = get_Landmarks(img, 'res')
        # print(img,gtLandmarks.shape)
        # print(img,resLandmarks.shape)
        ''' 
        68 points,contour [0:16],brow [17:26],nose [27:35],eye [36:45], mouth [46:67]
        gt:75 points,contour [10:26],brow [27:36],nose [37:45],eye [46:55], mouth [56:77]
        '''

        if 'corners' in normalization:
            # rsme_all
            rsme = np.mean(np.sqrt(np.sum((gtLandmarks - resLandmarks) ** 2, axis=1)))
            rsme_mean.append(rsme)
            information.append("{:.3f}".format(rsme))  # "{:.3f}".format() = str()
            #corner_all
            normDist_corners = np.linalg.norm(gtLandmarks[36] - gtLandmarks[45])
            error_corners = np.mean(np.sqrt(np.sum((gtLandmarks - resLandmarks) ** 2, axis=1))) / normDist_corners
            errors_corners.append(error_corners)
            information.append("{:.3f}".format(error_corners))
            #center_all
            normDist_centers = np.linalg.norm(np.mean(gtLandmarks[36:42], axis=0) - np.mean(gtLandmarks[42:48], axis=0))
            error_centers = np.mean(np.sqrt(np.sum((gtLandmarks - resLandmarks) ** 2, axis=1))) / normDist_centers
            errors_centers.append(error_centers)
            information.append("{:.3f}".format(error_centers))
            #diagonal_all
            height, width = np.max(gtLandmarks, axis=0) - np.min(gtLandmarks, axis=0)
            normDist_diagonal = np.sqrt(width ** 2 + height ** 2)
            error_diagonal = np.mean(np.sqrt(np.sum((gtLandmarks - resLandmarks) ** 2, axis=1))) / normDist_diagonal
            errors_diagonal.append(error_diagonal)
            information.append("{:.3f}".format(error_diagonal))

            # rsme_coutour
            rsme_coutour = np.mean(np.sqrt(np.sum((gtLandmarks[0:16] - resLandmarks[0:16]) ** 2, axis=1)))
            # print(gtLandmarks[1],gtLandmarks[0])
            rsme_coutour_mean.append(rsme_coutour)
            information.append("{:.3f}".format(rsme_coutour))  # 存入information
            #corner_coutour
            errors_corners_coutour = np.mean(np.sqrt(np.sum((gtLandmarks[0:16] - resLandmarks[0:16]) ** 2, axis=1))) / normDist_corners
            information.append("{:.3f}".format(errors_corners_coutour))
            #center_coutour
            errors_centers_coutour = np.mean(np.sqrt(np.sum((gtLandmarks[0:16] - resLandmarks[0:16]) ** 2, axis=1))) / normDist_centers
            information.append("{:.3f}".format(errors_centers_coutour))
            #diagonal_coutour
            errors_diagonal_coutour = np.mean(np.sqrt(np.sum((gtLandmarks[0:16] - resLandmarks[0:16]) ** 2, axis=1))) / normDist_diagonal
            information.append("{:.3f}".format(errors_diagonal_coutour))

            # rsme_brow
            rsme_brow = np.mean(np.sqrt(np.sum((gtLandmarks[17:26] - resLandmarks[17:26]) ** 2, axis=1)))
            rsme_brow_mean.append(rsme_brow)
            information.append("{:.3f}".format(rsme_brow))
            #corner_brow
            errors_corners_brow = np.mean(np.sqrt(np.sum((gtLandmarks[17:26] - resLandmarks[17:26]) ** 2, axis=1))) / normDist_corners
            information.append("{:.3f}".format(errors_corners_brow))
            #center_brow
            errors_centers_brow = np.mean(np.sqrt(np.sum((gtLandmarks[17:26] - resLandmarks[17:26]) ** 2, axis=1))) / normDist_centers
            information.append("{:.3f}".format(errors_centers_brow))
            #diagonal_brow
            errors_diagonal_brow = np.mean(np.sqrt(np.sum((gtLandmarks[17:26] - resLandmarks[17:26]) ** 2, axis=1))) / normDist_diagonal
            information.append("{:.3f}".format(errors_diagonal_brow))

            # rsme_nose
            rsme_nose = np.mean(np.sqrt(np.sum((gtLandmarks[27:35] - resLandmarks[27:35]) ** 2, axis=1)))
            rsme_nose_mean.append(rsme_nose)
            information.append("{:.3f}".format(rsme_nose))
            #corner_nose
            errors_corners_nose = np.mean(np.sqrt(np.sum((gtLandmarks[27:35] - resLandmarks[27:35]) ** 2, axis=1))) / normDist_corners
            information.append("{:.3f}".format(errors_corners_nose))
            #center_nose
            errors_centers_nose = np.mean(np.sqrt(np.sum((gtLandmarks[27:35] - resLandmarks[27:35]) ** 2, axis=1))) / normDist_centers
            information.append("{:.3f}".format(errors_centers_nose))
            #diagonal_nose
            errors_diagonal_nose = np.mean(np.sqrt(np.sum((gtLandmarks[27:35] - resLandmarks[27:35]) ** 2, axis=1))) / normDist_diagonal
            information.append("{:.3f}".format(errors_diagonal_nose))

            # rsme_eye
            rsme_eye = np.mean(np.sqrt(np.sum((gtLandmarks[36:45] - resLandmarks[36:45]) ** 2, axis=1)))
            rsme_eye_mean.append(rsme_eye)
            information.append("{:.3f}".format(rsme_eye))
            #corner_eye
            errors_corners_eye = np.mean(np.sqrt(np.sum((gtLandmarks[36:45] - resLandmarks[36:45]) ** 2, axis=1))) / normDist_corners
            information.append("{:.3f}".format(errors_corners_eye))
            #center_eye
            errors_centers_eye = np.mean(np.sqrt(np.sum((gtLandmarks[36:45] - resLandmarks[36:45]) ** 2, axis=1))) / normDist_centers
            information.append("{:.3f}".format(errors_centers_eye))
            #diagonal_eye
            errors_diagonal_eye = np.mean(np.sqrt(np.sum((gtLandmarks[36:45] - resLandmarks[36:45]) ** 2, axis=1))) / normDist_diagonal
            information.append("{:.3f}".format(errors_diagonal_eye))

            # rsme_mouth
            rsme_mouth = np.mean(np.sqrt(np.sum((gtLandmarks[46:67] - resLandmarks[46:67]) ** 2, axis=1)))
            rsme_mouth_mean.append(rsme_mouth)
            information.append("{:.3f}".format(rsme_mouth))
            #corner_mouth
            errors_corners_mouth = np.mean(np.sqrt(np.sum((gtLandmarks[46:67] - resLandmarks[46:67]) ** 2, axis=1))) / normDist_corners
            # errors_corners_mouth.append(errors_corners_mouth)
            information.append("{:.3f}".format(errors_corners_mouth))
            #center_mouth
            errors_centers_mouth = np.mean(np.sqrt(np.sum((gtLandmarks[46:67] - resLandmarks[46:67]) ** 2, axis=1))) / normDist_centers
            information.append("{:.3f}".format(errors_centers_mouth))
            #diagonal_mouth
            errors_diagonal_mouth = np.mean(np.sqrt(np.sum((gtLandmarks[46:67] - resLandmarks[46:67]) ** 2, axis=1))) / normDist_diagonal
            information.append("{:.3f}".format(errors_diagonal_mouth))


            # 修改保存数字格式
        if 'centers' in normalization:
            normDist_centers = np.linalg.norm(np.mean(gtLandmarks[36:42], axis=0) - np.mean(gtLandmarks[42:48], axis=0))
            error_centers = np.mean(np.sqrt(np.sum((gtLandmarks - resLandmarks) ** 2, axis=1))) / normDist_centers
            errors_centers.append(error_centers)

        elif 'diagonal' in normalization:
            height, width = np.max(gtLandmarks, axis=0) - np.min(gtLandmarks, axis=0)
            normDist_diagonal = np.sqrt(width ** 2 + height ** 2)
            error_diagonal = np.mean(np.sqrt(np.sum((gtLandmarks - resLandmarks) ** 2, axis=1))) / normDist_diagonal
            errors_diagonal.append(error_diagonal)

        one_information.append(information)

    two_information.append([gt_dir, "{:.3f}".format(np.mean(rsme_mean)),"{:.3f}".format(np.mean(errors_corners)),
                            "{:.3f}".format(np.mean(errors_centers)),"{:.3f}".format(np.mean(errors_diagonal)),
                            "{:.3f}".format(np.mean(rsme_coutour_mean)),"{:.3f}".format(np.mean(errors_corners_coutour)),
                            "{:.3f}".format(np.mean(errors_centers_coutour)),"{:.3f}".format(np.mean(errors_diagonal_coutour)),
                            "{:.3f}".format(np.mean(rsme_brow_mean)),"{:.3f}".format(np.mean(errors_corners_brow)),
                            "{:.3f}".format(np.mean(errors_centers_brow)),"{:.3f}".format(np.mean(errors_diagonal_brow)),
                            "{:.3f}".format(np.mean(rsme_nose_mean)),"{:.3f}".format(np.mean(errors_corners_nose)),
                            "{:.3f}".format(np.mean(errors_centers_nose)),"{:.3f}".format(np.mean(errors_diagonal_nose)),
                            "{:.3f}".format(np.mean(rsme_eye_mean)),"{:.3f}".format(np.mean(errors_corners_eye)),
                            "{:.3f}".format(np.mean(errors_centers_eye)),"{:.3f}".format(np.mean(errors_diagonal_eye)),
                            "{:.3f}".format(np.mean(rsme_mouth_mean)),"{:.3f}".format(np.mean(errors_corners_mouth)),
                            "{:.3f}".format(np.mean(errors_centers_mouth)),"{:.3f}".format(np.mean(errors_diagonal_mouth))])      #"{:.3f}".format() = str()

    write_excel("./results/img_infor", one_information, one_tittle)
    write_excel("./results/dir_infor", two_information, two_tittle)
    return errors_corners,errors_centers,errors_diagonal

def AUCError(errors, failureThreshold, step=0.0001, showCurve=True, file=None):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))

    ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]

    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]

    with open(file, "w") as f:
        for index in ced:
            f.write(str(index)+"\n")
    print("AUC @ {0}: {1}".format(failureThreshold, AUC))
    print("Failure rate: {0}".format(failureRate))
    plt.plot(xAxis, ced)
    if showCurve:
        plt.show()
    plt.savefig('AUC.png')

def main():
    img_list = []
    for file in os.listdir(gt_dir):
        if file.endswith('png') or file.endswith('jpg'):
            img_list.append(file)
    errors_corners,errors_centers,errors_diagonal = LandmarkError(img_list, normalization=['centers', 'corners','diagonal'])
    # errors_corners = LandmarkError(img_list, normalization=['corners'])
    failureThreshold = 0.35  # 可修改x轴最大值
    AUCError(errors_corners, failureThreshold, file="./results/XM2VTS_SDU_corners.txt")
    AUCError(errors_centers, failureThreshold, file="./results/XM2VTS_SDU_centers.txt")
    AUCError(errors_diagonal, failureThreshold, file="./results/XM2VTS_SDU_diagonal.txt")

if __name__ == '__main__':
    main()
