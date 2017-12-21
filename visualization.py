import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def setup_plot(ax):
    if ax == None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    return fig, ax


def plot_measurements(measurements_all, ax=None, cmap=get_cmap('Blues')):
    fig, ax = setup_plot(ax)
    time = np.array([measurements[0].timestamp for measurements in measurements_all])
    interval = (time-time[0])/(1.*time[-1]-time[0])
    for index, timestamp in enumerate(time):
        color = cmap(interval[index])
        [ax.plot(z.value[1], z.value[0], 'o',markeredgewidth=0.5, markeredgecolor="black", color=color) for z in measurements_all[index]]
    return fig, ax


def plot_track_pos(track_file, ax=None, color='k', add_index=False):
    fig, ax = setup_plot(ax)
    for track_id, state_list in track_file.items():
        states = np.array([est.est_posterior for est in state_list])
        ax.plot(states[:,2], states[:,0], color=color, label="Track_ID: " + str(track_id))
        ax.plot(states[0,2], states[0,0], 'o', color=color)
        if add_index:
            ax.text(states[-1,2], states[-1,0], str(track_id))
    return fig, ax
