from __future__ import with_statement
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({'font.size': 18})
from pylab import rcParams
rcParams['figure.figsize'] = 10, 4

SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def get_color(i):
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'black']
    return colors[i]

def get_marker(i):
    markers = ['|', 'v', '^', '+', 'x', 's', 'o']
    return markers[i]

def get_linestyle(i):
    linestyles = ['-', '--', '-.', ':']
    return linestyles[i % len(linestyles)]

def get_patterns(i):
    patterns = [ "/" , "x", "\\" , ".",  "|" , "-" , "+" ,  "o", "O",  "*" ]
    return patterns[i % len(patterns)]

def autolabel(rects, ax):
    # Get y-axis height to calculate label position from.
    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom

    for rect in rects:
        height = rect.get_height()

        # Fraction of axis height taken up by this rectangle
        p_height = (height / y_height)

        # If we can fit the label above the column, do that;
        # otherwise, put it inside the column.
        # if p_height > 0.95: # arbitrary; 95% looked good to me.
        #     label_position = height - (y_height * 0.05)
        # else:
        label_position = height + (y_height * 0.01)

        ax.text(rect.get_x() + rect.get_width()/2., label_position,
                '%0.2f' % height,
                ha='center', va='bottom')

def autolabelv1(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

filename = 'ssd-latency'

with open('D:\Research\ANN\Results\\' + filename +'.csv') as f:
    configs = f.readline().replace('\n', '').split(',')[1:]
    data = np.loadtxt(f, delimiter=',', dtype=float)
    y = data[:, 1:] # Remove the first column
    x_labels = data[:, :1]
    x_labels = [str(int(x[0])) for x in x_labels]

    num_x_axis_items = len(x_labels)

    num_config = len(y[0])
    locs = np.arange(1, num_x_axis_items + 1)
    width = 0.35

    fig, ax = plt.subplots()

    ax.margins(0.06, 0)

    for i in range(num_config):
        # plt.plot(x, y[:, i:i+1], label=configs[i])
        if i == num_config - 1:
            rect = ax.bar(locs+i*width, y[:, i], width=width, label=configs[i], color=get_color(i), edgecolor=get_color(i), linewidth=2)
        else:
            rect = ax.bar(locs+i*width, y[:, i], width=width, label=configs[i], color='none', edgecolor=get_color(i), linewidth=2, fill=False, hatch=get_patterns(i)*4)
        # rect = ax.bar(locs + i * width, y[i, :], width=width)
        # autolabel(rect, ax)
    # plt.grid(linestyle='dotted', linewidth='1')
    # plt.xticks('Text Similarity', 'ASR', 'BiDAF', locs)
    # plt.xlabel('Hidden dimension size')
    plt.ylim([0, 7])
    plt.xticks(locs + width * 1.0, x_labels[0:num_x_axis_items], fontsize=20);
    plt.ylabel('Execution time (ms)', fontsize=20, fontweight='bold')
    legend = plt.legend(bbox_to_anchor=(0.4,1.00), loc='upper center', ncol=2, frameon=False)
    # plt.legend()
    plt.gca().yaxis.grid(linestyle='-', color='grey', linewidth='0.4', dashes=(8, 6))
    plt.savefig('D:\Research\ANN\Results\ssd-latency.png')
    plt.show()
