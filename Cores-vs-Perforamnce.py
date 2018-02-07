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
rcParams['figure.figsize'] = 5, 4

batchsize1 = '1'
title1 = 'batch size ' + batchsize1
batchsize2 = '10'
title2 = 'batch size ' + batchsize2
title = '(b) GRU batching latency and throughput'
yaxis_legend1 = 'Execution time (ms)'
yaxis_legend2 = 'GFlops/sec'
xaxis_legend = 'Input/hidden dimension size'
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


datafilename11 = 'Cores-vs-Perforamnce'
datafilename_output = 'Core-vs-performance'


fig = plt.figure()
fig.subplots_adjust(hspace=0.27, wspace=0.22)
ax = fig.add_subplot(111)
ax11 = fig.add_subplot(111)

# ax.spines['top'].set_color('none')
# ax.spines['bottom'].set_color('none')
# ax.spines['left'].set_color('none')
# ax.spines['right'].set_color('none')
# ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

def get_color(i):
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black']
    return colors[i]

def get_marker(i):
    markers = ['|', 'v', '^', '+', 'x', 's', 'o']
    return markers[i]

def get_linestyle(i):
    linestyles = ['-', '--', '-.', ':']
    return linestyles[i % len(linestyles)]

with open(datafilepath + datafilename11 + '.csv') as f1:
    configs = f1.readline().replace('\n', '').split(',')[1:]
    data = np.loadtxt(f1, delimiter=',', dtype=float)
    x = data[:, :1]
    y = data[:, 1:] # Remove the first column
    num_config = len(y[0])
    for i in range(num_config):
        ax11.plot(x, y[:, i:i+1], label=configs[i], color=get_color(i), linestyle=get_linestyle(i), marker=get_marker(i),
     markerfacecolor=get_color(i), linewidth=2.5, markersize=8)

    ax11.grid(linestyle='dashed', linewidth='1.0', dashes=(8, 6))
    # ax11.set_xticks([32, 64, 128, 256, 512, 1024])
    ax11.set_xticks(np.arange(2, 22, 2))
    # ax11.set_xticks(x, [1, 2, 4, 8, 10, 12, 14, 16, 18, 20])
    # ax11.set_xscale('symlog', basex=2)
    # ax11.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax11.set_xscale('log', basex=2)
    ax11.set_yticks(np.arange(0, np.amax(y), 100))
    # ax11.set_yscale('symlog', basex=10)
    # ax11.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax11.set_xlabel('Threads', fontweight='bold', fontsize=16)
    ax11.set_ylabel('Gflops', fontweight='bold', fontsize=16)
    # ax1.set_title(title1)
    # ax11.legend(loc=2)
    # ax1.set_title(title1 + '/' + title2)

plt.legend(bbox_to_anchor=(0.5, 0.98), loc='upper center', ncol=2, prop={'size': 12})
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_useOffset(False)
# ax.set_title(title)
plt.savefig(datafilepath + datafilename_output + '.png', bbox_inches='tight')
plt.show()