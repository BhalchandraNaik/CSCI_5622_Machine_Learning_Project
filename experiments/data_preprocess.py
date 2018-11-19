import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def list2numpy(points_list,size=1):

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

    return X[:,:,1] == 255

df = pd.read_csv('C:\\Users\\bnaik\\Desktop\\CSCI_5622_Machine_Learning_Project\\data\\train_simplified\\bee.csv')
list2numpy(df.drawing[7],size=4)