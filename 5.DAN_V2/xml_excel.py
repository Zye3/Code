# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:06:36 2018

@author: libei
"""
# coding:utf-8
import os
import xlwt
import lxml
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

try:
    import xml.etree.CElementTree as ET
except:
    import xml.etree.ElementTree as ET


# 读xml文件
def xml_file(path, filetype):
    res = []
    for root, directory, files in os.walk(path):
        for filename in files:
            name, suf = os.path.splitext(filename)
            if suf == filetype:
                res.append(os.path.join(root, filename))
            else:
                continue
    return (res)


def xml_timebase(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    sequence = root[0]
    for i in range(len(sequence)):
        if sequence[i].tag == 'rate':
            rate = sequence[i]
            for i in range(len(rate)):
                if rate[i].tag == 'timebase':
                    timebase = rate[i].text
                    return (timebase)


def xml_duration(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    sequence = root[0]
    for i in range(len(sequence)):
        if sequence[i].tag == 'duration':
            duration = sequence[i].text
            return (duration)


# xml文件信息提取
def xml_read(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    sequence = root[0]
    duration = xml_duration(file_path)
    timebase = xml_timebase(file_path)
    times = time_frame(duration, timebase)
    # for k in sequence:
    medias = sequence.findall('media')
    for i in medias:
        videos = i.findall('video')
        for i in videos:
            tracks = i.findall('track')
            video_information = []
            for i in range(len(tracks)):
                track = tracks[i]
                orbital_name = 'track' + str(i)
                for j in range(len(track)):
                    if track[j].tag == 'clipitem':
                        clipitem = track[j]
                        list_1 = []
                        list_2 = []
                        file_video_path = '0'
                        for i in range(len(clipitem)):
                            list_2.append(clipitem[i].tag)
                            if clipitem[i].tag == 'start':
                                timeline_in = clipitem[i].text
                                timeline_in_string = time_frame(timeline_in, timebase_line)
                            if clipitem[i].tag == 'end':
                                timeline_out = clipitem[i].text
                                timeline_out_string = time_frame(timeline_out, timebase_line)
                            if clipitem[i].tag == 'rate':
                                rate = clipitem[i]
                                for i in range(len(rate)):
                                    if rate[i].tag == 'timebase':
                                        timebase_line = rate[i].text
                            if clipitem[i].tag == 'name':
                                filename = clipitem[i].text
                            if clipitem[i].tag == 'in':
                                file_start = clipitem[i].text
                                if file_start == '-1':
                                    transitionitem = track[j - 1]
                                    for i in range(len(transitionitem)):
                                        if transitionitem[i].tag == 'start':
                                            cross_superimposition_strat = transitionitem[i].text
                                        if transitionitem[i].tag == 'end':
                                            cross_superimposition_end = transitionitem[2].text
                                        try:
                                            file_start = (float(cross_superimposition_strat) + float(
                                                cross_superimposition_end)) / 2
                                        except:
                                            continue
                            if clipitem[i].tag == 'out':
                                file_end = clipitem[i].text
                                if file_end == '-1':
                                    transitionitem = track[j + 1]
                                    for i in range(len(transitionitem)):
                                        if transitionitem[i].tag == 'start':
                                            cross_superimposition_strat = transitionitem[1].text
                                        if transitionitem[i].tag == 'end':
                                            cross_superimposition_end = transitionitem[2].text
                                        try:
                                            file_end = (float(cross_superimposition_strat) + float(
                                                cross_superimposition_end)) / 2
                                        except:
                                            continue
                            try:
                                times_string = int(file_end) - int(file_start)
                                times_line = time_frame(times_string, timebase_line)
                            except:
                                continue
                            if clipitem[i].tag == 'file':
                                file_ = clipitem[i]
                                for i in range(len(file_)):
                                    if file_[i].tag == 'pathurl':
                                        file_video_path = file_[i].text
                                    if file_[i].tag == 'timecode':
                                        timecode = file_[i]
                                        for i in range(len(timecode)):
                                            if timecode[i].tag == 'string':
                                                file_start_time_string = timecode[i].text
                                                file_start_time_value = time_string(file_start_time_string,
                                                                                    timebase_line)
                                                file_start_string = time_frame(
                                                    int(file_start) + file_start_time_value, timebase_line)
                                                file_end_string = time_frame(int(file_end) + file_start_time_value,
                                                                             timebase_line)
                                    if file_[i].tag == 'media':
                                        media = file_[i]
                                        for i in range(len(media)):
                                            if media[i].tag == 'video':
                                                video = media[i]
                                                for i in range(len(video)):
                                                    if video[i].tag == 'samplecharacteristics':
                                                        samplecharacteristics = video[i]
                                                        for i in range(len(samplecharacteristics)):
                                                            if samplecharacteristics[i].tag == 'width':
                                                                resolution_width = samplecharacteristics[i].text
                                                            if samplecharacteristics[i].tag == 'height':
                                                                resolution_height = samplecharacteristics[i].text
                                                            try:
                                                                resolution = str(resolution_width) + '*' + str(
                                                                    resolution_height)
                                                            except:
                                                                continue
                            if 'file' not in list_2:

                                for index in video_information:
                                    if filename == index[0]:
                                        file_video_path = index[1]
                                        resolution = index[3]
                                        file_start_time_string = index[10]
                                        file_start_time_value = time_string(file_start_time_string, timebase_line)
                                        file_start_string = time_frame(int(file_start) + file_start_time_value,
                                                                       timebase_line)
                                        file_end_string = time_frame(int(file_end) + file_start_time_value,
                                                                     timebase_line)

                        list_1.append(filename)
                        list_1.append(file_video_path)
                        list_1.append(timebase)
                        list_1.append(resolution)
                        list_1.append(orbital_name)
                        list_1.append(timebase_line)
                        list_1.append(timeline_in)
                        list_1.append(timeline_in_string)
                        list_1.append(timeline_out)
                        list_1.append(timeline_out_string)
                        list_1.append(file_start_time_string)
                        list_1.append(file_start)
                        list_1.append(file_start_string)
                        list_1.append(file_end)
                        list_1.append(file_end_string)
                        list_1.append(times_line)
                        list_1.append(times_string)
                        list_1.append(times)
                        video_information.append(list_1)

                    else:
                        continue
            return (video_information)


# 将00:00:00:00转成int
def time_string(value, timebase):
    value_list = value.split(':')
    time_value = ((int(value_list[0]) * 60 + int(value_list[1])) * 60 + int(value_list[2])) * int(timebase) + int(
        value_list[3])
    return (time_value)


# 将int转成00:00:00:00
def time_frame(duration, timebase):
    hour = 0
    mintue = 0
    second = 0
    zhenshu = 0
    duration = int(duration)
    timebase = int(timebase)
    if duration % (timebase * 3600) == 0:
        hour = duration / (timebase * 3600)
    else:
        hour = duration / (timebase * 3600)
        mintues = duration % (timebase * 3600)
        if mintues % (timebase * 60) == 0:
            mintue = duration / (timebase * 60)
        else:
            mintue = mintues / (timebase * 60)
            seconds = mintues % (timebase * 60)
            if seconds % (timebase) == 0:
                second = seconds / timebase
            else:
                second = seconds / timebase
                zhenshu = seconds % timebase
    time = '%02d:%02d:%02d:%02d' % (hour, mintue, second, zhenshu)
    return (time)


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

    row0 = [u'源文件名', u'文件地址', u'帧率', u'分辨率', u'轨道', u'时间线帧率', u'时间线入点帧',
            u'时间线入点时码', u'时间线出点帧', u'时间线出点时码', u'源文件起始时码', u'源文件入点帧',
            u'源文件入点时码', u'源文件出点帧', u'源文件出点时码', u'时间线时长', u'片段帧数', u'总时长']
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


input_file = sys.argv[1]
#input_file = r'C:\Users\Administrator\Desktop\xml-excel\time_line\time_line'
filetype = '.xml'
xml_files = xml_file(input_file, filetype)
for file_path in xml_files:
    print(file_path)
    video_information = xml_read(file_path)
    a = write_excel(file_path, video_information)
