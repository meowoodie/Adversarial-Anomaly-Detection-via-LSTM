import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True

def read_results(filepath):
    seq_lens      = []
    normal_accs   = []
    abnormal_accs = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            seq_len, normal_acc, abnormal_acc = line.strip("\n").split(" ")
            seq_lens.append(int(seq_len))
            normal_accs.append(float(normal_acc))
            abnormal_accs.append(float(abnormal_acc))
    return seq_lens, normal_accs, abnormal_accs

if __name__ == "__main__":
    seq_lens, normal_accs, abnormal_accs = read_results("result/gan_macys.txt")

    fig, ax = plt.subplots()

    ax.set_xlim(min(seq_lens), max(seq_lens))
    ax.set_ylim(0, 1.1)
    ax.yaxis.grid(linestyle='dotted') # horizontal lines

    ax.set_xlabel(r'detect at the $i^{th}$ event', fontsize=16)
    ax.set_ylabel(r'average accuracy', fontsize=16)

    line1 = ax.plot(seq_lens, abnormal_accs, linestyle="-", color='red')
    line2 = ax.plot(seq_lens, normal_accs, linestyle="-", color='gray')

    ax.legend(['Anomaly data', 'Random normal data'])

    plt.show()