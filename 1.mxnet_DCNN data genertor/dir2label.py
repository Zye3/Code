import os
import numpy as np
import json

path = "Face_data/MutilPIE-semifrontal"

result = "Face_data/labels.txt"
result_re = []

for i in os.listdir(path):
    label = ""
    if i.split(".")[-1] == "pts":
        with open("{}/{}".format(path, i), "r") as f:
            data = f.readlines()
        label += "{}/{}.jpg".format(path, i.split(".")[0])
        points = data[10:78] #save line of pts file
        x_min, x_max, y_min, y_max = 1000000, 0, 1000000, 0
        for m in points:
            m = m.strip().split(" ")
            label += "," + m[0]
            label += "," + m[1]
            if float(m[0]) < x_min:
                x_min = float(m[0])
            if float(m[0]) > x_max:
                x_max = float(m[0])
            if float(m[1]) < y_min:
                y_min = float(m[1])
            if float(m[1]) > y_max:
                y_max = float(m[1])
        label += "," + str(x_min) + "," + str(y_min) + "," + str(x_max) + "," + str(y_max)
        result_re.append(label)
with open(result, "w") as f:
    for i in result_re:
        f.write(i+"\n")


