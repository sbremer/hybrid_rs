import scripts
import numpy as np
import results.plots as lplot
import matplotlib.pyplot as plt

x = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])

rmses_before_cf = [1.087754031907789, 1.0160846375623303, 0.98665142617269153, 0.97008461264651891, 0.95548276401057464, 0.94619710626315723, 0.93968497677483842, 0.93577502698213877, 0.92726121437510778, 0.9262644087024261, 0.9204449429617646, 0.91781728984277833, 0.91217374683500207]
rmses_before_md = [1.0192836603476507, 0.98599805797949469, 0.9709866591656855, 0.95929785676813883, 0.95194307797323419, 0.94516962692602569, 0.93946204085755036, 0.93719878310698135, 0.93537048559133462, 0.93323056741747024, 0.93156402651392756, 0.92903375063842275, 0.92879267157427925]
rmses_after_cf = [1.021076703950677, 0.98035703988113665, 0.96063928442376301, 0.94704785177663953, 0.9361603903387743, 0.92952226341087552, 0.92311381151853977, 0.92156511356257098, 0.91546341303589829, 0.91247460881781906, 0.90886877066048422, 0.90522465545990827, 0.90072854008219905]
rmses_after_md = [1.0167254898540934, 0.97931326747856151, 0.96272001712716215, 0.95151233303799265, 0.94332759040901282, 0.93803478195928092, 0.93299317105428381, 0.93049292606708656, 0.92805724570640658, 0.92621986687158098, 0.9247406278421052, 0.92266567238770836, 0.92120691572818736]

rmses_after_hybrid = []

for x_at, y1, y2 in zip(x, rmses_after_cf, rmses_after_md):
    rmses_after_hybrid.append(y1 if x_at >=10 else y2)


# plt.plot(x, rmses_before_cf, label='CF')
# plt.plot(x, rmses_before_md, label='MD')
# plt.plot(x, rmses_after_hybrid, label='Hybrid')
# plt.show()

fig, ax = lplot.newfig(1.0)

ax.plot(x, rmses_before_cf, '-', label='CF')
ax.plot(x, rmses_before_md, '--', label='MD')
ax.plot(x, rmses_after_hybrid, ':', label='Hybrid')
ax.set_xlabel('\#Ratings / User')
ax.set_ylabel('RMSE')
ax.legend()
lplot.savefig('ratings_per_user')
# plt.show()
