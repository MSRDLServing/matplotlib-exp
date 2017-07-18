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

with open('D:\Desktop\FaRNN\\rnn-comparison-batchsize100.csv') as f:
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
    plt.xlabel('Hidden dimension size')
    plt.ylabel('Execution time (ms)')
    # plt.ylabel('Throughput (Gigaops/sec)')
    plt.title('Batch size 100')
    # plt.title('Batch size 10')
    plt.legend()
    plt.savefig('D:\Desktop\FaRNN\\rnn-comparison-batchsize100.png')
    plt.show()
