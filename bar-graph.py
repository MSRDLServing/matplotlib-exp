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
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def get_color(i):
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black']
    return colors[i]

def get_marker(i):
    markers = ['|', 'v', '^', '+', 'x', 's', 'o']
    return markers[i]

def get_linestyle(i):
    linestyles = ['-', '--', '-.', ':']
    return linestyles[i % len(linestyles)]

def get_patterns(i):
    patterns = [ "/" , ".", "x", "\\" , "|" , "-" , "+" ,  "o", "O",  "*" ]
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
                '%d' % int(height),
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

filename = 'rnn-model-perf-comparison'

with open('D:\Desktop\FaRNN\\' + filename +'.csv') as f:
    configs = f.readline().replace('\n', '').split(',')[1:]
    data = np.loadtxt(f, delimiter=',', dtype=float)
    y = data[:, 1:] # Remove the first column
    x_labels = data[:, 1:]

    num_config = len(y[0])
    locs = np.arange(1, num_config + 1)
    width = 0.27

    fig, ax = plt.subplots()

    num_x_axis_items = len(y[0])
    for i in range(num_config):
        # plt.plot(x, y[:, i:i+1], label=configs[i])
        rect = ax.bar(locs+i*width, y[:, i], width=width, label=configs[i], color=get_color(i), edgecolor='black', hatch=get_patterns(i))
        # rect = ax.bar(locs + i * width, y[i, :], width=width)
        autolabel(rect, ax)
    # plt.grid(linestyle='dotted', linewidth='1')
    # plt.xticks('Text Similarity', 'ASR', 'BiDAF', locs)
    # plt.xlabel('Hidden dimension size')
    plt.ylim([0, 140])
    plt.xticks(locs + width * 1.5, ('Text Similarity', 'ASR', 'BiDAF'), fontsize=20);
    plt.ylabel('Execution time (ms)', fontsize=20, fontweight='bold')
    legend = plt.legend(bbox_to_anchor=(0.5, 1.17), loc='upper center', ncol=3, frameon=False)
    # plt.legend()
    plt.gca().yaxis.grid(linestyle='dashed', linewidth='1.0', dashes=(8, 6))
    plt.savefig('D:\Desktop\FaRNN\\rnn-model-perf-comparison.png')
    plt.show()
