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
datafilename = 'hardware-perf-counter-results'
yaxis_legend = 'L2 Miss'
# xaxis_legend = 'Configurations'
# yaxis_legend = 'Throughput (Gigaops/sec)'

with open(datafilepath + datafilename + '.csv') as f:
    configs = f.readline().replace('\n', '').split(',')[1:]
    data = np.loadtxt(f, delimiter=',', dtype=float)

    y = data[1:] # Remove the first column
    num_config = len(y)
    locs = np.arange(1, num_config + 1)
    width = 0.27

    plt.plot(locs, y)
    plt.grid(linestyle='dotted', linewidth='1')
    plt.xticks(locs, ('Parallel-GEMM', 'TF/CNTK Fusion + Parallel-GEMM', 'MM-DAG + Fusion + PCP', 'MM-DAG + Fusion + PCP + WCS'), rotation=17)

    # plt.xscale('log', basex=2)
    # plt.xlabel(configs)
    plt.ylabel(yaxis_legend)
    plt.yscale('log', basex=2)
    plt.legend()
    plt.savefig(datafilepath + datafilename + '.png')
    plt.show()
