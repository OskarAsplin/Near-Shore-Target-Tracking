import numpy as np
import matplotlib.pyplot as plt
import trajectory_tools
import tracking
import simulation
import visualization


# Global constants
clutter_density = 1e-5
radar_range = 1000

# Initialized target
x0_1 = np.array([100, 4, 0, 5])
x0_2 = np.array([-100, -4, 0, -5])
cov0 = np.diag([10, 0.5, 10, 0.5])
v_max = 50

# Time for simulation
dt = 1
t_end = 100
time = np.arange(0, t_end, dt)
K = len(time)             # Num steps

# Kalman filter stuff
q = 0.25                  # Process noise strength squared
r = 50
H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
R = r*np.identity(2)      # Measurement covariance
F, Q = tracking.DWNAModel.model(dt, q)


#PDA constants
P_G = 0.99
P_D = 0.9

# Empty state and covariance vectors
x_true = np.zeros((2, 4, K))

# -----------------------------------------------

# Set up tracking system
radar = simulation.SquareRadar(radar_range, clutter_density, P_D, R)
gate = tracking.TrackGate(P_G, v_max)
target_model = tracking.DWNAModel(q)
PDAF_tracker = tracking.PDAFTracker(P_D, target_model, gate)
track_manager = tracking.Manager(PDAF_tracker)

# Generate target trajectory - random turns, constant velocity
traj_tools = trajectory_tools.TrajectoryChange()
for k, t in enumerate(time):
    if k == 0:
        x_true[0, :, k] = x0_1
        x_true[1, :, k] = x0_2.T
    else:
        x_true[0, :, k] = F.dot(x_true[0, :, k - 1])
        x_true[1, :, k] = F.dot(x_true[1, :, k - 1])
        x_true[0, :, k] = traj_tools.randomize_direction(x_true[0, :, k]).reshape(4)
        x_true[1, :, k] = traj_tools.randomize_direction(x_true[1, :, k]).reshape(4)

# Initialize tracks
first_est_1 = tracking.Estimate(0, x0_1, cov0, is_posterior=True, track_index=0)
first_est_2 = tracking.Estimate(0, x0_2, cov0, is_posterior=True, track_index=1)
track_manager.add_new_tracks([[first_est_1], [first_est_2]])

# Run tracking
measurements_all = []
for k, timestamp in enumerate(time):
    measurements = radar.generate_measurements([H.dot(x_true[0, :, k]), H.dot(x_true[1, :, k])], timestamp)
    measurements_all.append(measurements)
    track_manager.step(measurements)

# Plot
fig, ax = visualization.plot_measurements(measurements_all)
ax.plot(x_true[0, 2, :], x_true[0, 0, :], 'k', label='True trajectory 1')
ax.plot(x_true[0, 2, 0], x_true[0, 0, 0], 'ko')
ax.plot(x_true[1, 2, :], x_true[1, 0, :], 'k', label='True trajectory 2')
ax.plot(x_true[1, 2, 0], x_true[1, 0, 0], 'ko')
visualization.plot_track_pos(track_manager.track_file, ax, 'r')
ax.set_xlim(-radar_range, radar_range)
ax.set_ylim(-radar_range, radar_range)
ax.set_xlabel('East[m]')
ax.set_ylabel('North[m]')
ax.set_title('Track position with sample rate: 1/s')
ax.legend()

plt.show()
