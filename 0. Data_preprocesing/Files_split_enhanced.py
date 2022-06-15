# -*- coding: utf-8 -*-

import os
import shutil


def GetFileNameWithoutExtension(fileName: str):
    return '.'.join(fileName.split('.')[:-1])


def ReorderFileList(files: list):
    # change this function if you want to sort the files
    import random
    random.shuffle(files)
    return files


def GetBaseFiles(folder: str):
    files = list(set([GetFileNameWithoutExtension(os.path.basename(f))
                      for f in os.listdir(folder)]))
    files = [folder+'/'+f for f in files]
    ReorderFileList(files)
    if len(files) == 0:
        print("No files found in %s" % folder)
        return []

    filesInGroup = int(len(files)/10)
    if filesInGroup <= 0:
        print("Error: number of files in %s is less than 10" % folder)
        return []

    result = [files[i:i+filesInGroup]
              for i in range(0, len(files), filesInGroup)]

    if len(files) % 10 != 0:
        print("Warning: number of files is not a multiple of 10")
        rest = len(files) % 10
        result[9].append(files[-rest:])
    return result


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
                    shutil.copy(src_path, output_path+"/"+str(singleNum))
                else:
                    shutil.copy(src_path, output_path +
                                "/except_"+str(singleNum))
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
        MakeDirectory(output_path+"/except_"+str(i))
        MakeDirectory(output_path+"/"+str(i))
        if not CheckDirectoryIsEmpty(output_path+"/except_"+str(i)):
            print("Warning: output directory %s is not empty" %
                  output_path+"/except_"+str(i))
        if not CheckDirectoryIsEmpty(output_path+"/"+str(i)):
            print("Warning: output directory %s is not empty" %
                  output_path+"/"+str(i))

        CopyFilesSingle(input_path, output_path, i, fileGroups, extension)


def DivideIntoTenGroups(names: list):
    filesInGroup = int(len(names)/10)
    return [names[i:i+filesInGroup] for i in range(0, len(names), filesInGroup)]


def main(argv):
    # initial variables
    folders = argv[1:-1]
    output_path = argv[-1]
    folders = ["/home/zye/Desktop/0.original_data/FRGC", "/home/zye/Desktop/0.original_data/XM2VTS"]

    if len(folders) <= 1:
        print("Usage: %s <input_folders> <output_folders>" % argv[0])
        return 1

    first_input = list()
    input_set = list()


    # first_input = ("/home/zye/Desktop/0.original_data/300W")
    # input_set = ["/home/zye/Desktop/0.original_data/FRGC","/home/zye/Desktop/0.original_data/XM2VTS"]

    isFirst = True
    for folder in folders:
        # traverse and get all files
        files = GetBaseFiles(folder)
        if len(files) == 0:
            continue
        if not isFirst:
            first_input = list(folder, files) # split ten group
        else:
            input_set.append(list(folder, files))

    for group in first_input[1]:
        # here are ten groups
        others = list()
        for other in first_input[1]:
            if other == group:  # is current group
                others += other

        current = group
        for other in input_set:
            for group in other[1]:
                others += group

        CopyFiles(current, others, input_set, ["jpg", "pts"])
        # TODO: copy files (current, others)

    return 0


if '__main__' == __name__:
    import sys
    sys.exit(main(sys.argv))
