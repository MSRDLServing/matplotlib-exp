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
    colors = ['darkorange', 'deepskyblue', 'darkviolet', 'mediumseagreen', 'magenta', 'black']
    return colors[i]

def get_marker(i):
    markers = ['s', 'o', 'v', '^', '+',  's', 'x']
    return markers[i]

def get_linestyle(i):
    linestyles = ['-', '--', '-.', ':']
    return linestyles[i % len(linestyles)]

def get_patterns(i):
    patterns = [ "/" , "x", "none" , ".",  "|" , "-" , "+" ,  "o", "O",  "*" ]
    return patterns[i % len(patterns)]

def billions(x, pos):
    """Formatter for Y axis, values are in billions"""
    return '%1.fbn' % (x*1e-6)

matplotlib.rcParams.update({'font.size': 18})
from pylab import rcParams
rcParams['figure.figsize'] = 5, 4

SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


datafilepath = 'D:\Research\ANN\Pruning\\'
datafilename = 'impact-on-latency-varying-pruning-ratio'
yaxis_legend = 'Search time (ms)'
xaxis_legend = 'L'
# title = 'Batch size '
# yaxis_legend = 'Throughput (Gigaops/sec)'

with open(datafilepath + datafilename + '.csv') as f:
    fig, ax = plt.subplots()
    configs = f.readline().replace('\n', '').split(',')[1:]
    data = np.loadtxt(f, delimiter=',', dtype=float)
    x = data[:, :1]
    y = data[:, 1:] # Remove the first column

    num_config = len(y[0])
    for i in range(num_config):
        ax.plot(x, y[:, i:i+1], label=configs[i], color=get_color(i), linestyle=get_linestyle(0),
                 marker=get_marker(i),
                 markerfacecolor=get_color(i), linewidth=2.0, markersize=8)
    ax.grid(linestyle='dotted', linewidth='1')
    # plt.xticks([64, 128, 256, 512, 1024])
    # ax.set_xticks(x)
    # ax.set_xscale('symlog', basex=2)
    # ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel(xaxis_legend)
    ax.set_ylabel(yaxis_legend)
    pylab.xlim([10,110])
    pylab.ylim([0.05,0.45])
    # plt.title(title)
    # plt.legend()
    # legend_properties = {'weight':'bold'}
    legend = plt.legend(bbox_to_anchor=(0.52,0.42), loc='lower right', ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig(datafilepath + datafilename + '.png', bbox_inches='tight')
    plt.show()
