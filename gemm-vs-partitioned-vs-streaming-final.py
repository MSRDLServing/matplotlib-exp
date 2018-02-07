import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 14})

def get_color(i):
    colors = ['blue','green','red','cyan','magenta','black']
    return colors[i]

def get_marker(i):
    markers = ['|','v','^','+','x','s','o']
    return markers[i]

def get_linestyle(i):
    linestyles = ['-','--','-.',':']
    return linestyles[i % len(linestyles)]

def get_patterns(i):
    patterns = ["/",".","x","\\","|","-","+","o","O","*"]
    return patterns[i %len(patterns)]

filename = "gemm-vs-partitioned-vs-streaming-final"
fd = open('D:\Desktop\FaRNN\\' + filename +'.csv', 'r')

data = fd.readlines()
header = map(str.strip, data)[0].split(",")[1:]

configs = []
sgemm = []
partition = []
streaming = []
prev_config = None
for config in map(lambda t: t.strip().split(","), data[1:]):
    if prev_config is not None:
        if prev_config.startswith("1x") and config[0].startswith("10x"):
            #for i in range(3):
            configs.append("")
            sgemm.append(0)
            partition.append(0)
            streaming.append(0)
    configs.append(config[0])
    sgemm.append(float(config[1]))
    partition.append(float(config[2]))
    streaming.append(float(config[3]))
    prev_config = config[0]

results = []
results.append(sgemm)
results.append(partition)
results.append(streaming)

n_groups = 11

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.26

ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed', linewidth=0.25)

labels = ["Parallel-GEMM", "PCP", "PCP + WCS"]

for i in range(3):
    rects1 = plt.bar(index + bar_width*i, results[i], bar_width,
                     color=get_color(i),
                     edgecolor='black',
                     hatch=get_patterns(i),
                     label=labels[i])

plt.ylabel('Gflops', fontweight='bold', fontsize=16)
# plt.title('Partitioning vs. Streamlining')
matplotlib.rcParams.update({'font.size': 12})
plt.xticks(index + bar_width / 2, configs, rotation=45)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig('D:\Desktop\FaRNN\\' + filename +'.png')
plt.show()
