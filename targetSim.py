import numpy as np
import matplotlib.pyplot as plt
import trajectory_tools
import tracking
import simulation


# Global constants
clutter_density = 0.0001
radar_range = 1500

# Initialized target
x0 = np.array([[-300], [4], [-200], [5]])
# cov0 = np.diag([10, 0.5, 10, 0.5])
# x0_est = x0 + np.dot(np.linalg.cholesky(cov0), np.random.normal(size=[4, 1]))
v_max = 50

# Time for simulation
dt = 1
t_end = 400
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
# gamma_g = 9.21        # Gate threshold

# Empty state and covariance vectors
x_true = np.zeros((4, K))
# cov_prior = np.zeros((4, 4, K))
# cov_posterior = np.zeros((4, 4, K))


radar = simulation.SquareRadar(radar_range, clutter_density, P_D, R)
gate = tracking.TrackGate(P_G, v_max)
target_model = tracking.DWNAModel(q)
PDAF_tracker = tracking.PDAFTracker(P_D, target_model, gate)
track_manager = tracking.Manager(PDAF_tracker)

# -----------------------------------------------

# Generate target trajectory - random turns, constant velocity
traj_tools = trajectory_tools.TrajectoryChange()
for k, t in enumerate(time):
     if k == 0:
         x_true[:, k] = x0.T
     else:
         x_true[:, k] = F.dot(x_true[:, k - 1])
         x_true[:, k] = traj_tools.randomize_direction(x_true[:, k]).reshape(4)

# Run tracking
measurements_all = []
for k, timestamp in enumerate(time):
     measurements = radar.generate_measurements([H.dot(x_true[:, k])], timestamp)
     measurements_all.append(measurements)
     track_manager.step(measurements)

x_est_posterior = track_manager.ret_posterior()
z_gate = track_manager.ret_measurements()

# Plot
plt.plot(x_true[2, :], x_true[0, :], 'r', label='True trajectory')
plt.plot(z_gate[1, :], z_gate[0, :], 'kx', label='Data within gate')
plt.plot(x_est_posterior[2, :], x_est_posterior[0, :], 'b--', label='Posterior estimation', linewidth=2)

plt.xlabel('East[m]')
plt.ylabel('North[m]')
plt.title('Track position with sample rate: 1/s')
plt.legend()

plt.show()