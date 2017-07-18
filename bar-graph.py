from __future__ import with_statement
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({'font.size': 10})

with open('D:\Desktop\FaRNN\\rnn-model-perf-comparison.csv') as f:
    configs = f.readline().replace('\n', '').split(',')[1:]
    data = np.loadtxt(f, delimiter=',', dtype=float)
    y = data[:, 1:] # Remove the first column
    num_config = len(y[0])
    locs = np.arange(1, num_config + 1)
    width = 0.27

    for i in range(num_config):
        # plt.plot(x, y[:, i:i+1], label=configs[i])
        plt.bar(locs+i*width, y[:, i], width=width, label=configs[i])
    # plt.grid(linestyle='dotted', linewidth='1')
    # plt.xticks('Text Similarity', 'ASR', 'BiDAF', locs)
    # plt.xlabel('Hidden dimension size')
    plt.xticks(locs + width * 1.5, ('Text Similarity', 'ASR', 'BiDAF'));
    plt.ylabel('Execution time (ms)')
    plt.legend()
    plt.gca().yaxis.grid(linestyle='dotted', linewidth='1.5')
    plt.savefig('D:\Desktop\FaRNN\\rnn-model-perf-comparison.png')
    plt.show()
