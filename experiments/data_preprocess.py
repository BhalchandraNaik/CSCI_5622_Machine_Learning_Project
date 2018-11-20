import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import ast

def list2numpy(points_list,size=2):

    """
    Takes a list of points and converts it to a boolean
    numpy array of size 72 by 72. Increase size to
    double the output size.
    """
    
    fig, ax = plt.subplots(figsize=(size,size))
    fig.tight_layout(pad=0)
    ax.grid(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_axis_off()


    for points in points_list:
        ax.set_xlim(0,255)
        ax.set_ylim(0,255)
        ax.invert_yaxis()
        ax.plot(points[0],points[1])

    fig.canvas.draw()

    X = np.array(fig.canvas.renderer._renderer)
    plt.close()

    return (~(X[:,:,1] == 255)).astype(int).flatten()

def get_class_name(file_path):
    i = len(file_path)-5
    name = ''
    while file_path[i]!='\\':
        name = file_path[i]+name
        i = i-1
    return name

def count_lines(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i-1

def read_csvs(csv_files, nrows=1000):
    df =  pd.concat([ pd.read_csv(file,nrows=nrows) for file in csv_files])
    df.reset_index(inplace=True,drop=True)
    df['drawing'] = df.drawing.apply(ast.literal_eval)
    return df

# Get all the files paths in the target folder
data_dir = 'C:\\Users\\bnaik\\Desktop\\CSCI_5622_Machine_Learning_Project\\data\\temp_data\\'
data_files = glob(data_dir+"*.csv")

# get the class names
class_names = [get_class_name(file_path) for file_path in data_files]
# print(class_names)

# Classes and data-points
line_counts = [count_lines(file_path) for file_path in data_files]
counts = pd.DataFrame({"line_counts":line_counts})
counts["class"] = class_names
# print(counts)

# Read the data into a dataframe
df = read_csvs(data_files, nrows = 10000)

# Now convert the timestamped vector into an actually drawing

# plt.imshow(list2numpy(df.drawing[7],size=4),cmap="gray")
# plt.axis('off')
# plt.show()
df['drawing'] = df['drawing'].apply(list2numpy)
df['drawing'] = df['drawing'].apply(np.array2string, threshold=40001)
df.to_csv("C:\\Users\\bnaik\\Desktop\\CSCI_5622_Machine_Learning_Project\\data\\my_data_.csv")