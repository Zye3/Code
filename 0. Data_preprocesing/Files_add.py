# -*- coding: utf-8 -*-

import os
import shutil


def GetFileNameWithoutExtension(fileName: str):
    """
    Function: get a base filename without extension
    :param fileName: tail part of folder, os.path.basename(f)
    :return: Base filesName without extension, split('.')[:-1]
    """
    return '.'.join(fileName.split('.')[:-1])


def GetBaseFiles(folder: str):
    """
    Function: get 10 groups filename
    :param folder: input folder of absolute file_path
    :return:result: 10 group files, result = [[(len(filesInGroup),index(filesInGroup))],...]
    """

    # get a list of all filesname from input folder
    files = list(set([GetFileNameWithoutExtension(os.path.basename(f))
                      for f in os.listdir(folder)]))

    # get files length of each group
    # filesInGroup = int(len(files) / 10)
    filesInGroup = len(files)

    if filesInGroup <= 0:
        print("Error: number of files is %s is less than 10" % folder)
        return []

    # save files into each group, based on the files length,
    # result = [[(len(filesInGroup),index(filesInGroup))],...]
    # result = [[files[i:i + filesInGroup]] for i in range(0, len(files), filesInGroup)]


    # if len(files) == 0:
    #     print("No files found in %s" % folder)
    #     return []
    # # if files can not  multiple by 10, then save the rest of files into the final group
    # if len(files) % 10 != 0:
    #     print("Warning:number of files is not a multiple of 10")
    #     rest_files = len(files) % 10
    #     result[9].append(rest_files[-rest_files:])

    return [files[i:i+filesInGroup] for i in range(0, len(files), filesInGroup)] # result


def ReorderFileList(files: list):
    # change this function if you want to sort the files
    """
    :param files: 10 groups files
    :return: random sort of files
    """
    import random
    random.shuffle(files)
    return files


def MakeDirectory(path: str):
    """
    :param path: path name.
    :return: create a folder, based on path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def CopyFilesSingle(input_path: str, output_path: str, singleNum: int, fileGroups: list, extension):
    """
    Function copy a group file to the output_path folder
    :param input_path: input_path folder
    :param output_path: the i-th out_path sub folder
    :param singleNum: the i-th number group files
    :param fileGroups: 10 groups filename
    :param extension: extension [.jpg,pts]
    :return:
    """
    index = 0
    # file name in each group
    for fileName_list in fileGroups:
        for baseName in fileName_list:
            # create a list of file name with a full input_path in each group
            src_paths = []
            # add extension to the tail part of file name and input_path tp the head part of files name
            if type(extension) is type([]):
                for ext in extension:
                    src_paths.append(input_path + "/" + baseName + "." + ext)
            elif type(extension) is type(""):
                src_paths.append(input_path + "/" + baseName + "." + extension)

            # copy file of full input_path in each group to output_path
            for src_path in src_paths:
                if not os.path.exists(src_path):
                    print("Error: file %s does not exist" % src_path)
                    continue
                if os.path.isdir(src_path):
                    print("Error: file %s is a directory" % src_path)
                    continue

                # copy the i-th number of group file name to output path.
                # copy((input_path),(output_path))
                if (singleNum == index):
                    shutil.copy(src_path, output_path + "/Train_set_" + str(singleNum))
                # # copy the i-th number of group filename to the other output path
                # else:
                #     shutil.copy(src_path, output_path +
                #                 "/except_" + str(singleNum))
        index += 1



def CheckDirectoryIsEmpty(path: str):
    """
    Function: Check if the path is empty ot not.
    :param path: input a path
    :return: Ture or Flase
    """
    if not os.path.exists(path):
        return True
    if len(os.listdir(path)) == 0:
        return True
    return False


def CopyFiles(input_path: str, output_path: str, fileGroups: list, extension):
    """

    :param input_path: input_path folder
    :param output_path: out_path folder
    :param fileGroups: 10 groups filename
    :param extension: extension
    :return:
    """
    # create a folder of output_path
    MakeDirectory(output_path)
    # create sub folders of output_path and copy file to each sub folder
    for i in range(10):
        MakeDirectory(output_path + "/Train_set_" + str(i))
        MakeDirectory(output_path + "/Test_set_" + str(i))
        if not CheckDirectoryIsEmpty(output_path + "/Train_set_" + str(i)):
            print("Warning: output directory %s is not empty" %
                  output_path + "/Train_set_" + str(i))
        if not CheckDirectoryIsEmpty(output_path + "/" + str(i)):
            print("Warning: output directory %s is not empty" %
                  output_path + "/Test_set_" + str(i))

        CopyFilesSingle(input_path, output_path, i, fileGroups, extension)


def main(argv):
    # if (len(argv) != 3):
    #     print("Usage: %s <input_path> <output_path>" % argv[0])
    #     return 1
    # input_path = argv[1]
    # output_path = argv[2]
    folders = ['/home/zye/Desktop/0.original_data/COFW', '/home/zye/Desktop/0.original_data/FRGC',
               '/home/zye/Desktop/0.original_data/XM2VTS', '/home/zye/Desktop/0.original_data/UCLAN']
    # input_path = "./fegc/fegc"
    output_path = "/home/zye/Desktop/2.Data_split_processed/300W_train_processed"
    for input_path in folders:
        files = GetBaseFiles(input_path)
        if len(files) == 0:
            return 1
        # default: random
        files = ReorderFileList(files)
        # extension could be list or string
        # using list is faster because there is less cycles
        CopyFiles(input_path, output_path, files, ["jpg", "pts"])
    # CopyFiles(input_path, output_path, files, "jpg")
    # CopyFiles(input_path, output_path, files, "pts")

    return 0


if '__main__' == __name__:
    import sys

    sys.exit(main(sys.argv))
