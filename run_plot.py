import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import visualization

true_IPDA_arr = [0.8805, 0.942, 0.948, 0.966, 0.985, 0.9815, 0.9885, 0.9895, 0.9925]
false_IPDA_arr = [0.0015, 0.0065, 0.0365, 0.0775, 0.3205, 0.565, 0.802, 0.8965, 0.9465]
true_MofN_arr =  [0.4195, 0.484, 0.531, 0.8415, 0.8835, 0.918, 0.9665, 0.9725, 0.9805]
false_MofN_arr = [0.014, 0.016, 0.0245, 0.1285, 0.24, 0.45, 0.8575, 0.983, 1.0]



# Plot
fig, ax = visualization.setup_plot(None)
plt.semilogx(false_IPDA_arr, true_IPDA_arr, '-', label='IPDA')
plt.semilogx(false_MofN_arr, true_MofN_arr, '-', label='M of N')
ax.set_title('ROC')
ax.set_xlabel(r'$P_{FT}$')
ax.set_ylabel(r'$P_{DT}$')
ax.legend()
# for axis in [ax.xaxis, ax.yaxis]:
#     axis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.show()


# # Plot false tracks
# xIPDA = (5e-06, 1e-05, 1.5e-05, 2e-05, 2.5e-05, 3e-05, 3.5e-05, 4e-05)
# yIPDA = (0, 0, 3, 5, 10, 29, 57, 93)
# xMofN = (5e-06, 1e-05, 1.5e-05, 2e-05, 2.5e-05, 3e-05, 3.5e-05, 4e-05)
# yMofN = (1, 12, 79, 273, 474, 818, 1088, 1335)
#
# fig, ax = visualization.setup_plot(None)
# plt.semilogy(xMofN, yMofN, '--', label='M of N')
# plt.semilogy(xIPDA, yIPDA, label='IPDA')
# ax.set_title('False tracks detected over 1000 scans')
# ax.set_xlabel('Clutter density')
# ax.set_ylabel('False tracks detected')
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# ax.legend()
# # plt.yscale('symlog')
# # for axis in [ax.xaxis, ax.yaxis]:
# #     axis.set_major_locator(ticker.MaxNLocator(integer=True))
# plt.show()



# # Plot all ROC curves together
# true_MofN_arr_6 = [0.556, 0.6605, 0.8345, 0.936]
# false_MofN_arr_6 = [0.022, 0.079, 0.2515, 0.725]
# true_MofN_arr_8 = [0.4145, 0.768, 0.8615, 0.912, 0.9475, 0.955]
# false_MofN_arr_8 = [0.0105, 0.0855, 0.2015, 0.4305, 0.835, 0.9215]
# true_MofN_arr_10 = [0.34, 0.43, 0.5085, 0.786, 0.877, 0.9075, 0.9635, 0.9715, 0.979]
# false_MofN_arr_10 = [0.0055, 0.0085, 0.0185, 0.066, 0.18, 0.3475, 0.7215, 0.9525, 0.995]
# true_MofN_arr_12 = [0.4195, 0.484, 0.531, 0.8415, 0.8835, 0.918, 0.9665, 0.9725, 0.9805]
# false_MofN_arr_12 = [0.014, 0.016, 0.0245, 0.1285, 0.24, 0.45, 0.8575, 0.983, 1.0]
#
# true_IPDA_arr_6 = [0.4065, 0.573, 0.659, 0.728, 0.7715, 0.8085, 0.8775, 0.8995, 0.913]
# false_IPDA_arr_6 = [0.0, 0.002, 0.009, 0.0205, 0.1445, 0.2875, 0.4795, 0.6165, 0.698]
# true_IPDA_arr_8 = [0.6625, 0.775, 0.854, 0.8605, 0.9015, 0.919, 0.9425, 0.9545, 0.9685]
# false_IPDA_arr_8 = [0.0005, 0.004, 0.02, 0.0385, 0.205, 0.401, 0.6225, 0.7615, 0.8325]
# true_IPDA_arr_10 = [0.8185, 0.887, 0.9115, 0.941, 0.957, 0.964, 0.9785, 0.982, 0.981]
# false_IPDA_arr_10 = [0.0015, 0.009, 0.0265, 0.054, 0.2475, 0.5035, 0.7385, 0.842, 0.9015]
# true_IPDA_arr_12 = [0.8805, 0.942, 0.948, 0.966, 0.985, 0.9815, 0.9885, 0.9895, 0.9925]
# false_IPDA_arr_12 = [0.0015, 0.0065, 0.0365, 0.0775, 0.3205, 0.565, 0.802, 0.8965, 0.9465]
#
#
#
#
# # Plot
# fig, ax = visualization.setup_plot(None)
# plt.semilogx(false_IPDA_arr_6, true_IPDA_arr_6, 'C0:', label='IPDA: 6 scans')
# plt.semilogx(false_IPDA_arr_8, true_IPDA_arr_8, 'C0-.', label='IPDA: 8 scans')
# plt.semilogx(false_IPDA_arr_10, true_IPDA_arr_10, 'C0--', label='IPDA: 10 scans')
# plt.semilogx(false_IPDA_arr_12, true_IPDA_arr_12, 'C0-', label='IPDA: 12 scans')
# plt.semilogx(false_MofN_arr_6, true_MofN_arr_6, 'C1:', label='M of N: 6 scans')
# plt.semilogx(false_MofN_arr_8, true_MofN_arr_8, 'C1-.', label='M of N: 8 scans')
# plt.semilogx(false_MofN_arr_10, true_MofN_arr_10, 'C1--', label='M of N: 10 scans')
# plt.semilogx(false_MofN_arr_12, true_MofN_arr_12, 'C1-', label='M of N: 12 scans')
# ax.set_title('ROC')
# ax.set_xlabel(r'$P_{FT}$')
# ax.set_ylabel(r'$P_{DT}$')
# ax.legend()
# # for axis in [ax.xaxis, ax.yaxis]:
# #     axis.set_major_locator(ticker.MaxNLocator(integer=True))
# plt.ylim([0, 1])
# plt.xlim([0, 1])
# plt.show()


# # Plot
# fig, ax = visualization.setup_plot(None)
# plt.semilogx(false_IPDA_arr_6, true_IPDA_arr_6, 'C0-', label='IPDA')
# plt.semilogx(false_IPDA_arr_8, true_IPDA_arr_8, 'C0-')
# plt.semilogx(false_IPDA_arr_10, true_IPDA_arr_10, 'C0-')
# plt.semilogx(false_IPDA_arr_12, true_IPDA_arr_12, 'C0-')
# plt.semilogx(false_MofN_arr_6, true_MofN_arr_6, 'C1-', label='M of N')
# plt.semilogx(false_MofN_arr_8, true_MofN_arr_8, 'C1-')
# plt.semilogx(false_MofN_arr_10, true_MofN_arr_10, 'C1-')
# plt.semilogx(false_MofN_arr_12, true_MofN_arr_12, 'C1-')
# ax.set_title('ROC')
# ax.set_xlabel(r'$P_{FA}$')
# ax.set_ylabel(r'$P_D$')
# ax.legend()
# # for axis in [ax.xaxis, ax.yaxis]:
# #     axis.set_major_locator(ticker.MaxNLocator(integer=True))
# plt.ylim([0, 1])
# plt.xlim([0, 1])
# plt.show()
