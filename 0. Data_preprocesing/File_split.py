# -*- coding: utf-8 -*-

import os
import shutil


def GetFileNameWithoutExtension(fileName: str):
    return '.'.join(fileName.split('.')[:-1])


def GetBaseFiles(folder: str):
    files = list(set([GetFileNameWithoutExtension(os.path.basename(f))
                      for f in os.listdir(folder)]))
    if len(files) == 0:
        print("No files found in %s" % folder)
        return []
    if len(files) % 10 != 0:
        print("Error: number of files is not a multiple of 10")
        return []

    filesInGroup = int(len(files)/10)
    return [files[i:i+filesInGroup] for i in range(0, len(files), filesInGroup)]


def ReorderFileList(files: list):
    # change this function if you want to sort the files
    import random
    random.shuffle(files)
    return files


def MakeDirectory(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def CopyFilesSingle(input_path: str, output_path: str, singleNum: int, fileGroups: list, extension):
    index = 0
    for fileGroup in fileGroups:
        for baseName in fileGroup:
            src_paths = []
            if type(extension) is type([]):
                for ext in extension:
                    src_paths.append(input_path+"/"+baseName+"."+ext)
            elif type(extension) is type(""):
                src_paths.append(input_path+"/"+baseName+"."+extension)

            for src_path in src_paths:
                if not os.path.exists(src_path):
                    print("Error: file %s does not exist" % src_path)
                    continue
                if os.path.isdir(src_path):
                    print("Error: file %s is a directory" % src_path)
                    continue
                if (singleNum == index):
                    shutil.copy(src_path, output_path+"/Test_set_"+str(singleNum))
                else:
                    shutil.copy(src_path, output_path +
                                "/Train_set_"+str(singleNum))
        index += 1


def CheckDirectoryIsEmpty(path: str):
    if not os.path.exists(path):
        return True
    if len(os.listdir(path)) == 0:
        return True
    return False


def CopyFiles(input_path: str, output_path: str, fileGroups: list, extension):
    MakeDirectory(output_path)
    for i in range(10):
        MakeDirectory(output_path+"/Train_set_"+str(i))
        MakeDirectory(output_path+"/Test_set_"+str(i))
        if not CheckDirectoryIsEmpty(output_path+"/Train_set_"+str(i)):
            print("Warning: output directory %s is not empty" %
                  output_path+"/Train_set_"+str(i))
        if not CheckDirectoryIsEmpty(output_path+"/Test_set_"+str(i)):
            print("Warning: output directory %s is not empty" %
                  output_path+"Test_set/"+str(i))

        CopyFilesSingle(input_path, output_path, i, fileGroups, extension)


def main(argv):
    # if (len(argv) != 3):
    #     print("Usage: %s <input_path> <output_path>" % argv[0])
    #     return 1
    # input_path = argv[1]
    # output_path = argv[2]
    input_path = "0.original_data/MutilPLE_semifrontal_train"
    output_path = "2.Data_split_processed/MutilPLE_semifrontal_train_processed"
    files = GetBaseFiles(input_path)
    if len(files) == 0:
        return 1
    # default: random
    files = ReorderFileList(files)
    # extension could be list or string
    # using list is faster because there is less cycles
    CopyFiles(input_path, output_path, files, ["jpg","pts"])
    # CopyFiles(input_path, output_path, files, "jpg")
    # CopyFiles(input_path, output_path, files, "pts")
    return 0


if '__main__' == __name__:
    import sys
    sys.exit(main(sys.argv))
