import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['text.usetex'] = True

def read_results(filepath):
    seq_lens      = []
    normal_accs   = []
    abnormal_accs = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            print(line.strip("\n").split("  "))
            seq_len, normal_acc, abnormal_acc = line.strip("\n").split("  ")
            seq_lens.append(int(seq_len))
            normal_accs.append(float(normal_acc))
            abnormal_accs.append(float(abnormal_acc))
    return seq_lens, normal_accs, abnormal_accs

if __name__ == "__main__":
    # filename = "gan_earthquake"
    filename = "pca_earthquake"
    seq_lens, normal_accs, abnormal_accs = read_results("result/%s.txt" % filename)

    print(normal_accs)

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    with PdfPages("result/%s.pdf" % filename) as pdf:
        fig, ax = plt.subplots()

        ax.set_xlim(min(seq_lens), max(seq_lens))
        ax.set_ylim(0, 1.1)
        ax.yaxis.grid(linestyle='dotted') # horizontal lines

        ax.set_xlabel(r'detect at the $i^{th}$ event', fontsize=20)
        ax.set_ylabel(r'average accuracy', fontsize=20)

        line1 = ax.plot(seq_lens, abnormal_accs, linestyle="-", color='blue')
        line2 = ax.plot(seq_lens, normal_accs, linestyle="-", color='gray')

        ax.legend(['Anomaly data', 'Random normal data'], fontsize=15, loc="bottom right")

        # plt.show()
        pdf.savefig(fig)