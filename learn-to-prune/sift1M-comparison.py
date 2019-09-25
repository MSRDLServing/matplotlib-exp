from __future__ import with_statement
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pylab

import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter
import matplotlib.font_manager as font_manager

def get_color(i):
    colors = ['darkviolet', 'darkviolet', 'deepskyblue', 'deepskyblue', 'darkorange', 'darkorange', 'darkorange', 'mediumseagreen']
    return colors[i]

def get_marker(i):
    markers = ['v', 'v', 'o', 'o', '^',  '^', '^', 's']
    return markers[i]

def get_linestyle(i):
    linestyles = ['-', '--', '-', '--', '-', '--', '-.', '-']
    return linestyles[i % len(linestyles)]

def get_patterns(i):
    patterns = [ "/" , "x", "none" , ".",  "|" , "-" , "+" ,  "o", "O",  "*" ]
    return patterns[i % len(patterns)]

def billions(x, pos):
    """Formatter for Y axis, values are in billions"""
    return '%1.fbn' % (x*1e-6)

def billions(x, pos):
    """Formatter for Y axis, values are in billions"""
    return '%1.fbn' % (x*1e-6)

matplotlib.rcParams.update({'font.size': 18})
from pylab import rcParams
rcParams['figure.figsize'] = 10, 8

SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 18
EXTRA_BIGGER_SIZE = 24

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=EXTRA_BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=EXTRA_BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


datafilepath = 'D:\OneDrive\Research\ANN\Pruning\\'
datafilename = 'sift1M-comparison'
yaxis_legend = 'Search time (ms)'
xaxis_legend = 'Accuracy (Top-1 recall)'
# title = 'Batch size '
# yaxis_legend = 'Throughput (Gigaops/sec)'

with open(datafilepath + datafilename + '.csv') as f:
    fig, ax = plt.subplots()
    # configs = f.readline().replace('\n', '').split(',')[1:]
    data = np.loadtxt(f, delimiter=',', dtype=float)
    config = data[:, :1]
    values = data[:, 1:] # Remove the first column

    x = values[:, 0::2]
    y = values[:, 1::2]

    configs = ["EFANNA", "EFANNA-sparse", "NSG", "NSG-sparse", "HNSW", "HNSW-sparse", "HNSW-rand", "APG"]

    num_config = len(x[0])
    for i in range(num_config):
        ax.plot(x[:, i:i+1], y[:, i:i+1], label=configs[i], color=get_color(i), linestyle=get_linestyle(i),
                 marker=get_marker(i),
                 markerfacecolor=get_color(i), linewidth=2.5, markersize=10)
    ax.grid(linestyle='dotted', linewidth='1')
    # plt.xticks([64, 128, 256, 512, 1024])
    # ax.set_xticks(x)
    # ax.set_xscale('symlog', basex=2)
    # ax.xaxis.set_major_formatter(ScalarFormatter())
    # plt.xticks(x, my_xticks)
    ax.set_xlabel(xaxis_legend)
    ax.set_ylabel(yaxis_legend)
    pylab.xlim([0.80,1])
    pylab.ylim([0,0.6])
    # plt.title(title)
    # plt.legend()
    plt.gca().invert_xaxis()

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0,2,4,6,1,3,5,7]
    # ["EFANNA-dense",  "NSG-dense", "HNSW-dense", "HNSW-rand", "EFANNA-sparse", "NSG-sparse", "HNSW-sparse", "APG"]
    # ["EFANNA-dense" 0 , "EFANNA-sparse" 1, "NSG-dense" 2, "NSG-sparse" 3, "HNSW-dense" 4, "HNSW-sparse" 5, "HNSW-rand" 6, "APG" 7]
    legend_properties = {'weight':'bold'}

    legend = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], bbox_to_anchor=(1.0,0.74), loc='lower right', ncol=2, frameon=False, prop=legend_properties)
    plt.tight_layout()
    plt.savefig(datafilepath + datafilename + '.png', bbox_inches='tight')
    plt.show()
