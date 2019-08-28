from __future__ import with_statement
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pylab

def get_color(i):
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'black']
    return colors[i]

def get_marker(i):
    markers = ['v', 'o', '|',  '^', '+', 'x', 's']
    return markers[i]

def get_linestyle(i):
    linestyles = ['--', '-', '-.', ':']
    return linestyles[i % len(linestyles)]

def get_patterns(i):
    patterns = [ "/" , "x", "\\" , ".",  "|" , "-" , "+" ,  "o", "O",  "*" ]
    return patterns[i % len(patterns)]

matplotlib.rcParams.update({'font.size': 18})
from pylab import rcParams
rcParams['figure.figsize'] = 5, 4

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

datafilepath = 'D:\Research\ANN\Results\\'
datafilename = 'sift1m-latency-large-k-varying-cluster'
yaxis_legend = 'Execution time (ms)'
xaxis_legend = 'Number of selected clusters'

with open(datafilepath + datafilename + '.csv') as f:
    fig, ax = plt.subplots()
    configs = f.readline().replace('\n', '').split(',')[1:]
    data = np.loadtxt(f, delimiter=',', dtype=float)
    x = data[:, :1]
    y = data[:, 1:] # Remove the first column

    x = np.array([0,1,2,3, 4])
    my_xticks = ['1', '4', '8', '32', '128']

    num_config = len(y[0])

    ts = y[:, 0]
    tc = y[:, 1]

    width = 0.5

    below = ax.bar(x, ts, width=width, label=configs[0], color='none', edgecolor=get_color(0), linewidth=2, fill=False, hatch=get_patterns(0)*4)
    above = ax.bar(x, tc, width=width, label=configs[1], color='none', edgecolor=get_color(1), linewidth=2, fill=False, hatch=get_patterns(1)*4, bottom=ts)

    ax.grid(linestyle='dotted', linewidth='1')
    # plt.xticks([64, 128, 256, 512, 1024])
    # ax.set_xticks(x)
    # ax.set_xscale('symlog', basex=2)
    # ax.xaxis.set_major_formatter(ScalarFormatter())
    plt.xticks(x + 0.25, my_xticks)
    ax.set_xlabel(xaxis_legend)
    ax.set_ylabel(yaxis_legend)
    pylab.xlim([-0.1,4.6])
    pylab.ylim([0,2.6])
    # plt.title(title)
    # plt.legend()
    legend_properties = {'weight':'bold'}
    legend = plt.legend(bbox_to_anchor=(0.04,0.95), loc='upper left', ncol=2, frameon=False, prop=legend_properties)
    plt.tight_layout()
    plt.savefig(datafilepath + datafilename + '.png', bbox_inches='tight')
    plt.show()


