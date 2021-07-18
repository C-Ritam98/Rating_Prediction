
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd

files = os.listdir()

file_name = "combined_data_"


#Reading reviews from file


with open(file_name + "1." + "txt",'r') as f:
          lines = f.read().split('\n')
          
reviews = []
movie_id = 0
new_row = []
lines = lines[:-1]

with open("train_data.csv",'w') as f:

    writer = csv.writer(f)
    writer.writerow(["Movie_Id","Person_Id","Rating"])

    for line in lines:
        if len(line)==0:
            continue
        if(line[-1] == ':'):
            #print(movie_id)
            movie_id += 1
            continue
        new_row = [movie_id] + line.split(',')[:-1]
        new_row[1] = int(new_row[1])
        if new_row[1]>10000:
            continue
        new_row[2] = int(new_row[2])
        #reviews.append(new_row)
        writer.writerow(new_row)

with open(file_name + "2." + "txt",'r') as f:
          lines = f.read().split('\n')
          
reviews = []
new_row = []
lines = lines[:-1]

with open("train_data.csv",'w') as f:

    writer = csv.writer(f)
    writer.writerow(["Movie_Id","Person_Id","Rating"])

    for line in lines:
        if len(line)==0:
            continue
        if(line[-1] == ':'):
            #print(movie_id)
            movie_id += 1
            continue
        new_row = [movie_id] + line.split(',')[:-1]
        new_row[1] = int(new_row[1])
        if new_row[1]>10000:
            continue
        new_row[2] = int(new_row[2])
        #reviews.append(new_row)
        writer.writerow(new_row)




