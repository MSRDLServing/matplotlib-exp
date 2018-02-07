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

matplotlib.rcParams.update({'font.size': 14})

from pylab import rcParams
rcParams['figure.figsize'] = 10, 4

batchsize1 = '1'
title1 = 'batch size ' + batchsize1
batchsize2 = '10'
title2 = 'batch size ' + batchsize2
title = '(b) GRU batching latency and throughput'
yaxis_legend1 = 'Execution time (ms)'
yaxis_legend2 = 'GFLOPS'

datafilepath = 'D:\Desktop\FaRNN\\'
datafilepath = 'D:\Desktop\FaRNN\\'
# datafilename1 = 'lstm-latency-opt-batch' + batchsize1
# datafilename2 = 'lstm-latency-opt-batch' + batchsize2
# datafilename12 = 'lstm-latency-opt-batch-' + batchsize1 + '-' + batchsize2

# datafilename1 = 'lstm-throughput-opt-batch' + batchsize1
# datafilename2 = 'lstm-throughput-opt-batch' + batchsize2
# datafilename12 = 'lstm-throughput-opt-batch-' + batchsize1 + '-' + batchsize2

# datafilename1 = 'gru-latency-opt-batch' + batchsize1
# datafilename2 = 'gru-latency-opt-batch' + batchsize2
# datafilename12 = 'gru-latency-opt-batch-' + batchsize1 + '-' + batchsize2

# datafilename1 = 'gru-throughput-opt-batch' + batchsize1
# datafilename2 = 'gru-throughput-opt-batch' + batchsize2
# datafilename12 = 'gru-throughput-opt-batch-' + batchsize1 + '-' + batchsize2

# datafilename1 = 'lstm-latency-batchN'
# datafilename2 = 'lstm-throughput-batchN'
# datafilename12 = 'lstm-latency-throughput-batchN'

xaxis_legend = 'Input/hidden dimension size'
datafilename1 = 'lstm-input-hidden-dim-size-avg-time'
datafilename2 = 'lstm-input-hidden-dim-size-throughput'
datafilename12 = 'lstm-input-hidden-dim-size-comparison'

# xaxis_legend = 'Sequence length'
# datafilename1 = 'lstm-seq-len-avg-time'
# datafilename2 = 'lstm-seq-len-throughput'
# datafilename12 = 'lstm-seq-len-comparison1'

# xaxis_legend = 'Batch size'
# datafilename1 = 'lstm-batch-avg-time'
# datafilename2 = 'lstm-batch-throughput'
# datafilename12 = 'lstm-batch-comparison1'

# datafilename1 = 'lstm-latency-opt-batch1'
# datafilename2 = 'lstm-throughput-opt-batch1'
# datafilename12 = 'lstm-latency-throughput-opt-batch1'

# datafilename1 = 'gru-latency-opt-batch1'
# datafilename2 = 'gru-throughput-opt-batch1'
# datafilename12 = 'gru-latency-throughput-opt-batch1'

fig = plt.figure()
fig.subplots_adjust(hspace=0, wspace=0.22)
ax = fig.add_subplot(111)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

def get_color(i):
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black']
    return colors[i]

def get_marker(i):
    markers = ['|', 'v', '^', '+', 'x', 's', 'o']
    return markers[i]

def get_linestyle(i):
    linestyles = ['-', '--', '-.', ':']
    return linestyles[i % len(linestyles)]

with open(datafilepath + datafilename1 + '.csv') as f1:
    configs = f1.readline().replace('\n', '').split(',')[1:]
    data = np.loadtxt(f1, delimiter=',', dtype=float)
    x = data[:, :1]
    y = data[:, 1:] # Remove the first column
    num_config = len(y[0])
    for i in range(num_config):
        ax1.plot(x, y[:, i:i+1], label=configs[i], color=get_color(i), linestyle=get_linestyle(i), marker=get_marker(i),
     markerfacecolor=get_color(i), linewidth=2.5, markersize=8)

    ax1.grid(linestyle='dotted', linewidth='1')

    # ax1.set_xticks(x)
    ax1.set_xticks(x, [32, 64, 128, 256, 512, 1024])
    ax1.set_xscale('symlog', basex=2)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax1.set_xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # ax1.set_xticks(np.arange(2, max(x) + 1, 2))

    # ax1.set_yticks(np.arange(0, np.amax(y), 5))
    ax1.set_yticks(x, [1, 10, 100])
    ax1.set_yscale('log', basex=10)
    ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.minorticks_off()

    ax1.set_xlabel(xaxis_legend)
    ax1.set_ylabel(yaxis_legend1)
    # ax1.set_title(title1)
    # ax1.legend(loc=2)
    # ax1.set_title(title1 + '/' + title2)

with open(datafilepath + datafilename2 + '.csv') as f2:
    configs = f2.readline().replace('\n', '').split(',')[1:]
    data = np.loadtxt(f2, delimiter=',', dtype=float)
    x = data[:, :1]
    y = data[:, 1:] # Remove the first column
    num_config = len(y[0])
    for i in range(num_config):
        ax2.plot(x, y[:, i:i + 1], label=configs[i], color=get_color(i), linestyle=get_linestyle(i),
                 marker=get_marker(i),
                 markerfacecolor=get_color(i), linewidth=2.5, markersize=8)
    ax2.grid(linestyle='dotted', linewidth='1')

    ax2.set_xticks(x, [32, 64, 128, 256, 512, 1024])
    ax2.set_xscale('symlog', basex=2)
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax2.set_xticks(x)
    # ax2.set_xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # ax2.set_xticks(np.arange(2, max(x) + 1, 2))

    ax2.set_yticks(np.arange(0, np.amax(y), 25))

    ax2.set_xlabel(xaxis_legend)
    ax2.set_ylabel(yaxis_legend2)
    # ax2.set_title(title2)

# ax.set_title(title)
# plt.legend(bbox_to_anchor=(-0.1, -0.45), loc='lower center', ncol=3)
plt.savefig(datafilepath + datafilename12 + '.png', bbox_inches='tight')
plt.show()