import scripts
import numpy as np
import results.plots as lplot
import matplotlib.pyplot as plt

# Normal
rmses_epochs_cf = [0.89960148840391463, 0.89779517729386416, 0.89672559940305019, 0.89626290401741215, 0.89578679881217804, 0.89561542943160377, 0.89567319920943567, 0.89551506213611276, 0.89570706473919093, 0.89575679537764541, 0.89575830777449195]
rmses_epochs_md = [0.9265575851917649, 0.92583061093396068, 0.92528327829188939, 0.92495449742979619, 0.92465154436997887, 0.92431301872002847, 0.92413590792169642, 0.92391674979954352, 0.92369720645138498, 0.92350945550720509, 0.92339056888259974]
rmses_epochs_hybrid = [0.89966098335769318, 0.89755639939507892, 0.8965226571710122, 0.89607192989109608, 0.89560595280242195, 0.89550923249699732, 0.89553063070340022, 0.89539816499084035, 0.89561169299303722, 0.89561603677956969, 0.89564290242563105]


rmses_epochs_cf = np.array(rmses_epochs_cf)
rmses_epochs_md = np.array(rmses_epochs_md)
rmses_epochs_hybrid = np.array(rmses_epochs_hybrid)

rmses_epochs_cf = rmses_epochs_cf - rmses_epochs_cf[0]
rmses_epochs_md = rmses_epochs_md - rmses_epochs_md[0]
rmses_epochs_hybrid = rmses_epochs_hybrid - rmses_epochs_hybrid[0]

x = range(len(rmses_epochs_cf))

# plt.plot(x, rmses_epochs_cf, '-', label='CF')
# plt.plot(x, rmses_epochs_md, '--', label='MD')
# plt.plot(x, rmses_epochs_hybrid, ':', label='Hybrid')
# plt.legend()
# plt.show()

fig, ax = lplot.newfig(1.0)

plt.style.use('acm-1col')
ax.plot(x, rmses_epochs_cf, '-', label='CF')
ax.plot(x, rmses_epochs_md, '--', label='MD')
ax.plot(x, rmses_epochs_hybrid, ':', label='Hybrid')
ax.set_xlabel('Cross-Training Epochs')
ax.set_ylabel('Change of RMSE')
ax.legend()
lplot.savefig('hybrid_epochs')
