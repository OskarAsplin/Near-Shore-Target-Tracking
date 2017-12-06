import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import visualization

true_IPDA_arr = [0.8185, 0.887, 0.9115, 0.941, 0.957, 0.964, 0.9785, 0.982, 0.981]
false_IPDA_arr = [0.0015, 0.009, 0.0265, 0.054, 0.2475, 0.5035, 0.7385, 0.842, 0.9015]

true_MofN_arr =  [0.34, 0.43, 0.5085, 0.786, 0.877, 0.9075, 0.9635, 0.9715, 0.979]
false_MofN_arr = [0.0055, 0.0085, 0.0185, 0.066, 0.18, 0.3475, 0.7215, 0.9525, 0.995]

# Plot
fig, ax = visualization.setup_plot(None)
plt.plot(false_IPDA_arr, true_IPDA_arr, '*-', label='IPDA')
plt.plot(false_MofN_arr, true_MofN_arr, '*-', label='M of N')
ax.set_title('ROC')
ax.set_xlabel(r'$P_{FA}$')
ax.set_ylabel(r'$P_D$')
ax.legend()
# for axis in [ax.xaxis, ax.yaxis]:
#     axis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.show()