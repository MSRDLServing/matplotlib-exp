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



batchsize1 = '1'
title1 = 'batch size ' + batchsize1
batchsize2 = '10'
title2 = 'batch size ' + batchsize2
title = 'GRU ' + title1 + ' and ' + title2
yaxis_legend = 'Throughput (Gigaops/sec)'
# yaxis_legend = 'Execution time (ms)'
xaxis_legend = 'Hidden dimension size'
datafilepath = 'D:\Desktop\FaRNN\\'
# datafilename1 = 'lstm-throughput-opt-batch' + batchsize1
# datafilename2 = 'lstm-throughput-opt-batch' + batchsize2
# datafilename12 = 'lstm-throughput-opt-batch-' + batchsize1 + '-' + batchsize2
# datafilename1 = 'gru-latency-opt-batch' + batchsize1
# datafilename2 = 'gru-latency-opt-batch' + batchsize2
# datafilename12 = 'gru-latency-opt-batch-' + batchsize1 + '-' + batchsize2
datafilename1 = 'gru-throughput-opt-batch' + batchsize1
datafilename2 = 'gru-throughput-opt-batch' + batchsize2
datafilename12 = 'gru-throughput-opt-batch-' + batchsize1 + '-' + batchsize2

fig = plt.figure()
ax = fig.add_subplot(111)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

with open(datafilepath + datafilename1 + '.csv') as f1:
    configs = f1.readline().replace('\n', '').split(',')[1:]
    data = np.loadtxt(f1, delimiter=',', dtype=float)
    x = data[:, :1]
    y = data[:, 1:] # Remove the first column
    num_config = len(y[0])
    for i in range(num_config):
        ax1.plot(x, y[:, i:i+1], label=configs[i])
    ax1.grid(linestyle='dotted', linewidth='1')
    # plt.xticks([64, 128, 256, 512, 1024])
    ax1.set_xscale('log', basex=2)
    ax1.set_yticks(np.arange(0, np.amax(y), 50))
    # ax1.set_xlabel('none')
    # ax1.set_ylabel('none')
    # ax1.set_title(title1)
    ax1.legend()
    # ax1.set_title(title1 + '/' + title2)

with open(datafilepath + datafilename2 + '.csv') as f2:
    configs = f2.readline().replace('\n', '').split(',')[1:]
    data = np.loadtxt(f2, delimiter=',', dtype=float)
    x = data[:, :1]
    y = data[:, 1:] # Remove the first column
    num_config = len(y[0])
    for i in range(num_config):
        ax2.plot(x, y[:, i:i+1], label=configs[i])
    ax2.grid(linestyle='dotted', linewidth='1')
    # plt.xticks([64, 128, 256, 512, 1024])
    ax2.set_xticks(x)
    ax2.set_xscale('log', basex=2)
    ax2.set_yticks(np.arange(0, np.amax(y), 100))
    # ax2.set_xlabel(xaxis_legend)
    # ax2.set_ylabel(yaxis_legend)
    # ax2.set_title(title2)

ax.set_title(title)
ax.set_xlabel(xaxis_legend)
ax.set_ylabel(yaxis_legend)
plt.savefig(datafilepath + datafilename12 + '.png')
plt.show()