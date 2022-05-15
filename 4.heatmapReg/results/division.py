import os


path = "./newpts"#predict path
path_1 = "./image"# GT path

files = [i.split(".")[0] for i in os.listdir(path)] #load path's folder filename before .


result = "other"
if os.path.isdir(result) == False: #if path don't exist the folder, create a new one
    os.mkdir(result)

for file in os.listdir(path_1): #load path's folder filename
    # if file.split(".")[0] in files: # file+_pred = files DCNN
    if file.split(".")[0]+"_pred" in files: #OpenFace
        continue
    else:
        os.rename("{}/{}".format(path_1, file), "{}/{}".format(result, file)) #file in patch_1 move to file in result

