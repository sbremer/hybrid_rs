import numpy as np
# import results.plots as lplot
import matplotlib.pyplot as plt

# # Normal
rmses_epochs_cf = [0.89871391966622272, 0.89801953407399149, 0.89637120263350345, 0.89622641955300364, 0.89645749534175057, 0.89679681842880066, 0.8970646800071076, 0.89739404569727077, 0.897932696039214, 0.89840521783122473, 0.89894268211972972]
rmses_epochs_md = [0.92768680982736773, 0.92508630348373833, 0.92395628519279782, 0.92358965970879559, 0.923284393281261, 0.92305838341505297, 0.92300587014004787, 0.92291819317069412, 0.92284853432111957, 0.92284402245768715, 0.92288891004958629]

# Coldstart
# rmses_epochs_cf = [1.0890348272091348, 1.0227211255627047, 1.0217580107038859, 1.0213056047008608, 1.0211928858521013, 1.0210529638365688, 1.0207690127529934, 1.0205373643498017, 1.0207205203293839, 1.020805172320383, 1.0202658436606722]
# rmses_epochs_md = [1.019422290642984, 1.0178845054797354, 1.0174474322784191, 1.0172457871871889, 1.0169962388874103, 1.0168267183304573, 1.0169104186621039, 1.0169892430200405, 1.0171037238296456, 1.0169569270858083, 1.0170247108552808]
# rmses_epochs_cf[0] = float('nan')

x = range(len(rmses_epochs_cf))

plt.plot(x, rmses_epochs_cf)
plt.plot(x, rmses_epochs_md)
plt.show()
