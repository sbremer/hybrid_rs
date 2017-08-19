import script_chdir
import numpy as np
import results.plots as lplot

recall_bias = np.array(
    [0, 0.1215, 0.2039, 0.2642, 0.3174, 0.3619, 0.4021, 0.4368, 0.4682, 0.4974
        , 0.5222, 0.5482, 0.5698, 0.5916, 0.6114, 0.6304, 0.647, 0.6647, 0.6802
        , 0.6947, 0.7082, 0.7193, 0.7299, 0.7405, 0.7517, 0.7616, 0.7695, 0.7773
        , 0.7855, 0.7933, 0.8007, 0.8082, 0.8155, 0.8214, 0.8275, 0.8331, 0.8383
        , 0.8434, 0.8484, 0.853, 0.8573, 0.8614, 0.8654, 0.8695, 0.8741, 0.8779
        , 0.8819, 0.8853, 0.8888, 0.8925, 0.8955, 0.8985, 0.9011, 0.9043, 0.9077
        , 0.9104, 0.913, 0.9155, 0.9185, 0.9208, 0.9226, 0.9246, 0.9266, 0.9288
        , 0.9309, 0.9336, 0.9358, 0.9387, 0.9409, 0.9434, 0.9465, 0.9487, 0.9511
        , 0.9535, 0.9554, 0.9581, 0.9598, 0.9625, 0.9648, 0.9667, 0.9685, 0.9707
        , 0.9722, 0.9742, 0.9761, 0.9776, 0.9792, 0.9812, 0.9829, 0.9843, 0.9861
        , 0.9872, 0.9884, 0.9896, 0.9913, 0.9926, 0.9938, 0.9953, 0.9962, 0.9975, 1])

recall_svdpp = np.array(
    [0, 0.196, 0.2983, 0.3715, 0.4277, 0.4736, 0.5138, 0.5489, 0.5797, 0.6083
        , 0.634, 0.6584, 0.6795, 0.6994, 0.7174, 0.7317, 0.7462, 0.7604, 0.7728
        , 0.7859, 0.7965, 0.8063, 0.8149, 0.8226, 0.829, 0.8363, 0.8432, 0.8499
        , 0.8567, 0.8627, 0.8686, 0.8733, 0.8783, 0.8832, 0.888, 0.8925, 0.8971
        , 0.9018, 0.9061, 0.910, 0.9133, 0.9174, 0.9212, 0.924, 0.9264, 0.9293
        , 0.9316, 0.934, 0.9361, 0.9386, 0.9409, 0.9431, 0.9457, 0.9483, 0.95
        , 0.9517, 0.9534, 0.9548, 0.9565, 0.958, 0.9596, 0.9614, 0.9625, 0.9638
        , 0.9654, 0.9666, 0.968, 0.969, 0.970, 0.9717, 0.9727, 0.9738, 0.9747
        , 0.9761, 0.9769, 0.9782, 0.9793, 0.9802, 0.9811, 0.9824, 0.9834, 0.984
        , 0.9849, 0.9855, 0.9867, 0.9874, 0.9883, 0.9891, 0.9896, 0.9903, 0.9918
        , 0.9928, 0.9935, 0.9944, 0.995, 0.9958, 0.9966, 0.9974, 0.9981, 0.9989, 1])

recall_sigmoidasym = np.array(
    [0, 0.2428, 0.3622, 0.443, 0.5065, 0.5573, 0.6029, 0.6393, 0.6695, 0.6963
        , 0.7198, 0.742, 0.7591, 0.7765, 0.7919, 0.805, 0.8171, 0.8285, 0.8398
        , 0.8491, 0.8575, 0.8657, 0.873, 0.8807, 0.8865, 0.8919, 0.897, 0.9023
        , 0.9072, 0.9114, 0.9148, 0.9194, 0.9232, 0.9265, 0.9293, 0.9322, 0.9351
        , 0.9381, 0.9407, 0.9427, 0.9449, 0.9472, 0.949, 0.9509, 0.9528, 0.9547
        , 0.9563, 0.958, 0.9593, 0.9608, 0.9623, 0.9636, 0.9647, 0.9658, 0.9669
        , 0.9677, 0.9689, 0.9697, 0.9708, 0.9715, 0.9725, 0.9732, 0.974, 0.9751
        , 0.9757, 0.9762, 0.977, 0.9775, 0.978, 0.9786, 0.9791, 0.9798, 0.9804
        , 0.981, 0.9817, 0.9827, 0.9831, 0.9837, 0.9842, 0.9846, 0.9853, 0.9861
        , 0.9866, 0.9875, 0.9884, 0.9893, 0.990, 0.9909, 0.9919, 0.9926, 0.9932
        , 0.9939, 0.9946, 0.9953, 0.9958, 0.9963, 0.9971, 0.9978, 0.9984, 0.9992, 1])

x = range(101)

fig, ax = lplot.newfig(1.0)

# ax.plot(x, recall_item_pop, '-', label='Item Popularity (AURC: {:.4f})'.format(np.mean(recall_item_pop[1:-1])))
ax.plot(x, recall_bias, '-', label='Bias Baseline (AURC: {:.4f})'.format(np.mean(recall_bias[1:-1])))
ax.plot(x, recall_svdpp, '--', label='SVD++ (AURC: {:.4f})'.format(np.mean(recall_svdpp[1:-1])))
ax.plot(x, recall_sigmoidasym, ':', label='SigmoidUserAsymFactoring (AURC: {:.4f})'.format(np.mean(recall_sigmoidasym[1:-1])))
ax.set_xlabel('N')
ax.set_ylabel('Recall(N)')
ax.legend()
lplot.savefig('recall')