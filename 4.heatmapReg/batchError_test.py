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

def get_Landmarks(img, gt_res):
    if gt_res == 'gt':
        # img = img.replace(img[-3:],'ptv')
        # img = os.path.join(gt_dir,img)
        # data = pd.read_csv(img,sep=',',header=None)
        # return data.values.astype(np.int64)
        img = img.replace(img[-4:], '.pts')
        img = os.path.join(gt_dir, img)
        with open(img, "r") as f:
             content = f.readlines()
        data = []


        # json pts file
        for i in content[3:-1]:#75points [10:-1]
            data.append([float(i.strip().split(" ")[0]), float(i.strip().split(" ")[1])])
        data = np.array(data, dtype=np.int64)
        # data = pd.read_csv(img,sep=' ',header=None)
        return data


    elif gt_res == 'res':
        img = img.replace(img[-4:],'_pred.pts')
        img = os.path.join(res_dir,img)

        with open(img, "r") as f:
             content = f.readlines()
        data = []

        # json pts file
        for i in content[3:-1]:
            data.append([float(i.strip().split(" ")[0]), float(i.strip().split(" ")[1])])
        data = np.array(data, dtype=np.int64)
        # data = pd.read_csv(img,sep=' ',header=None)
        return data


def LandmarkError(img_list, normalization=None, showResults=False, verbose=False):
    """
    :param img_list:
    :param normalization: must be a list
    :param showResults:
    :param verbose:
    :return:
    """
    errors_centers = []
    errors_corners = []
    errors = []
    rsme_mean = []

    # title
    one_tittle = ['imgname', 'rsme', 'NME-centers', 'NME-corners']
    one_information = []
    two_tittle = ['dirname', 'rsme', 'NME-centers', 'NME-corners','Failure Rate','AUC@(0.10)']
    two_information = []

    for i, img in enumerate(img_list):
        information = []
        information.append(img)
        gtLandmarks = get_Landmarks(img, 'gt')

        resLandmarks = get_Landmarks(img, 'res')
        #print(gtLandmarks.shape)
        # if normalization == 'centers':
        #     normDist = np.linalg.norm(np.mean(gtLandmarks[36:42], axis=0) - np.mean(gtLandmarks[42:48], axis=0))
        # elif normalization == 'corners':
        #     normDist = np.linalg.norm(gtLandmarks[36] - gtLandmarks[45])
        # elif normalization == 'diagonal':
        #     height, width = np.max(gtLandmarks, axis=0) - np.min(gtLandmarks, axis=0)
        #     normDist = np.sqrt(width ** 2 + height ** 2)
        rsme = np.mean(np.sqrt(np.sum((gtLandmarks - resLandmarks) ** 2, axis=1)))
        rsme_mean.append(rsme)
        # xiu gai bao cun shu zi ge shi
        information.append("{:.3f}".format(rsme))#"{:.3f}".format() = str()
        if 'centers' in normalization:
            normDist_centers = np.linalg.norm(np.mean(gtLandmarks[36:42], axis=0) - np.mean(gtLandmarks[42:48], axis=0))
            error_centers = np.mean(np.sqrt(np.sum((gtLandmarks - resLandmarks) ** 2, axis=1))) / normDist_centers
            errors_centers.append(error_centers)
            information.append("{:.3f}".format(error_centers))

        if 'corners' in  normalization:
            normDist_corners = np.linalg.norm(gtLandmarks[36] - gtLandmarks[45])
            error_corners = np.mean(np.sqrt(np.sum((gtLandmarks - resLandmarks) ** 2, axis=1))) / normDist_corners
            errors_corners.append(error_corners)
            information.append("{:.3f}".format(error_corners))

        elif 'diagonal' in normalization:
            height, width = np.max(gtLandmarks, axis=0) - np.min(gtLandmarks, axis=0)
            normDist_diagonal = np.sqrt(width ** 2 + height ** 2)
            error_diagonal = np.mean(np.sqrt(np.sum((gtLandmarks - resLandmarks) ** 2, axis=1))) / normDist_diagonal
            errors.append(error_diagonal)
        one_information.append(information)
        #print(normDist)
        # error = np.mean(np.sqrt(np.sum((gtLandmarks - resLandmarks) ** 2, axis=1))) / normDist
        # errors.append(error)
        # if verbose:
        #     print("{0}: {1}, {2}, {3}".format(i, error_centers, ))

        if showResults:
            plt.imshow(img[0], cmap=plt.cm.gray)
            plt.plot(resLandmarks[:, 0], resLandmarks[:, 1], 'o')
            plt.show()
    two_information.append([gt_dir, "{:.3f}".format(np.mean(rsme_mean)), "{:.3f}".format(np.mean(errors_centers)), "{:.3f}".format(np.mean(errors_corners))])#"{:.3f}".format() = str()

    write_excel("./results/img_infor", one_information, one_tittle)
    write_excel("./results/dir_infor", two_information, two_tittle)

    # if verbose:
    #     print("Image idxs sorted by error")
    #     print(np.argsort(errors))
    # avgError = np.mean(errors)
    # print("Average error : {0}".format(avgError))

    return errors_centers, errors_corners


def AUCError(errors, failureThreshold, step=0.0001, showCurve=True):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))

    ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]

    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]

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
    error_centers, errors_corners = LandmarkError(img_list,normalization=['centers','corners'])
    AUCError(error_centers, 0.10)
    AUCError(errors_corners, 0.10)

if __name__ == '__main__':
    main()
