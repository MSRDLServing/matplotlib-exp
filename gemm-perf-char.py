from __future__ import with_statement
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({'font.size': 14})

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
                '%s' % str(height),
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

filename = 'gemm-perf-char'

with open('D:\Desktop\FaRNN\\' + filename +'.csv') as f:
    configs = f.readline().replace('\n', '').split(',')[1:]
    data = np.loadtxt(f, delimiter=',', dtype=float)
    y = data[:, 1:] # Remove the first column
    x_labels = data[:, 1:]

    num_config = len(y[0])
    num_data_point = len(y)
    locs = np.arange(1, num_data_point + 1)
    width = 0.45

    fig, ax = plt.subplots()

    for i in range(num_config):
        # plt.plot(x, y[:, i:i+1], label=configs[i])
        rect = ax.bar(locs+i*width, y[:, i], width=width, label=configs[i], color=get_color(i), edgecolor='black', hatch=get_patterns(i))
        # rect = ax.bar(locs + i * width, y[i, :], width=width)
        # autolabel(rect, ax)
    # plt.grid(linestyle='dotted', linewidth='1')
    # plt.xticks('Text Similarity', 'ASR', 'BiDAF', locs)
    # plt.xlabel('Hidden dimension size')
    plt.ylim([0, 100])
    labels = ['4Kx4Kx4K',
              '.5Kx.5Kx.5K',
              '1Kx1Kx1K',
              '250x250x250',
              '125x125x125',
              '50x50x50',
              '1x1Kx1K',
              '2x1Kx1K',
              '4x1Kx1K',
              '8x1Kx1K',
              '16x1Kx1K',
              '64x1Kx1K',
              '256x1Kx1K'
              ]
    plt.xticks(locs + width * 1.5, labels);
    fig.autofmt_xdate()
    plt.ylabel('Percentage')
    plt.legend(bbox_to_anchor=(0.5, 0.98), loc='upper center', ncol=3)
    # plt.legend()
    plt.gca().yaxis.grid(linestyle='dashed', linewidth='1.0', dashes=(8, 6))
    plt.savefig('D:\Desktop\FaRNN\\'+filename+'.png')
    plt.show()
