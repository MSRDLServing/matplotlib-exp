from __future__ import with_statement
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as font_manager

def billions(x, pos):
    """Formatter for Y axis, values are in billions"""
    return '%1.fbn' % (x*1e-6)

matplotlib.rcParams.update({'font.size': 12})

datafilepath = 'D:\Desktop\FaRNN\\'
batchsize = '1'
datafilename = 'lstm-latency-opt-batch' + batchsize
yaxis_legend = 'Execution time (ms)'
xaxis_legend = 'Hidden dimension size'
title = 'Batch size ' + batchsize
# yaxis_legend = 'Throughput (Gigaops/sec)'

with open(datafilepath + datafilename + '.csv') as f:
    configs = f.readline().replace('\n', '').split(',')[1:]
    data = np.loadtxt(f, delimiter=',', dtype=float)
    x = data[:, :1]
    y = data[:, 1:] # Remove the first column
    num_config = len(y[0])
    for i in range(num_config):
        plt.plot(x, y[:, i:i+1], label=configs[i])
    plt.grid(linestyle='dotted', linewidth='1')
    # plt.xticks([64, 128, 256, 512, 1024])
    plt.xticks(x)
    plt.xscale('log', basex=2)
    plt.xlabel(xaxis_legend)
    plt.ylabel(yaxis_legend)
    plt.title(title)
    plt.legend()
    plt.savefig(datafilepath + datafilename + '.png')
    plt.show()
