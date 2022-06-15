import os
from PIL import Image


dirname_read="/home/zye/Desktop/0.original_data/UCLAN_ori/"
dirname_write="/home/zye/Desktop/0.original_data/UCLAN/"
names=os.listdir(dirname_read)
count=0
for name in names:
    img = Image.open(os.path.join(dirname_read,name))
    name = name.split(".")
    # print(name)
    if name[-1] == "png":
        name[-1] = "jpg"
        name = str.join(".", name)
        #r,g,b,a=img.split()
        #img=Image.merge("RGB",(r,g,b))
        to_save_path = dirname_write+name
        print(to_save_path)
        img.save(to_save_path)
        count+=1
        print(to_save_path, "------conut：",count)
    else:
        continue
