import numpy as np
import matplotlib as mpl

mpl.use('pgf')

import matplotlib.pyplot as plt

golden_mean = (np.sqrt(5.0) - 1.0) / 2.0


def savefig(filename):
    plt.tight_layout(pad=0.2)
    plt.savefig('plots/{}.pgf'.format(filename))
    plt.savefig('plots/{}.pdf'.format(filename))


def figsize(scale, ratio=golden_mean):
    fig_width_pt = 241.14749  # <- One column of ACM Conf. Get this from LaTeX using \the\linewidth
    # fig_width_pt = 422.52348 # Normal thesis template
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * ratio  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def newfig(width, ratio=golden_mean):
    plt.clf()
    fig = plt.figure(figsize=figsize(width, ratio))
    ax = fig.add_subplot(111)
    return fig, ax


pgf_with_latex = {  # setup matplotlib to use latex for output
    'pgf.texsystem': 'pdflatex',  # change this if using xetex or lautex
    'text.usetex': True,  # use LaTeX to write all text
    'font.family': 'serif',
    'font.serif': [],  # blank entries should cause plots to inherit fonts from the document
    'font.sans-serif': [],
    'font.monospace': [],
    'axes.labelsize': 10,  # LaTeX default is 10pt font.
    'font.size': 8,
    'legend.fontsize': 8,  # Make the legend/label fonts a little smaller
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    # 'figure.figsize': figsize(0.9),  # default fig size of 0.9 textwidth
    # 'figure.autolayout': True,
    'pgf.preamble': [
        r'\usepackage[utf8x]{inputenc}',  # use utf8 fonts becasue your computer can handle it :)
        r'\usepackage[T1]{fontenc}',  # plots will be generated using this preamble
    ]
}
mpl.rcParams.update(pgf_with_latex)
