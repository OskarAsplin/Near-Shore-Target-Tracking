import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import trajectory_tools
import tracking
import simulation
import visualization
import track_initiation
import analysis_sim


# Global constants
clutter_density = 2e-5
radar_range = 1000

# Initialized target
num_ships = 2
x0_1 = np.array([100, 4, 0, 5])
x0_2 = np.array([-100, -4, 0, -5])
x0 = [x0_1, x0_2]

# Time for simulation
dt = 1
t_end = 25
time = np.arange(0, t_end, dt)
K = len(time)             # Num steps

# Empty state vectors
x_true = np.zeros((num_ships, 4, K))

# Kalman filter stuff
q = 0.25                  # Process noise strength squared
r = 50                    # Measurement noise strength squared
H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
R = r*np.identity(2)      # Measurement covariance
F, Q = tracking.DWNAModel.model(dt, q)

#PDA constants
P_G = 0.99
P_D = 0.9

# -----------------------------------------------

# Define initiation and termination parameters
# IPDA
p11 = 0.98          # Survival probability
p21 = 0             # Probability of birth
P_Markov = np.array([[p11, 1 - p11], [p21, 1 - p21]])
initiate_thresh = 0.95
terminate_thresh = 0.10
# MofN
N_test = 6
M_req = 4
N_terminate = 3

c1 = 15
c2 = 25

# Set up tracking system
v_max = 10*dt
radar = simulation.SquareRadar(radar_range, clutter_density, P_D, R)
gate = tracking.TrackGate(P_G, v_max)
target_model = tracking.DWNAModel(q)

PDAF_tracker = tracking.PDAFTracker(P_D, target_model, gate)
M_of_N = track_initiation.MOfNInitiation(M_req, N_test, PDAF_tracker, gate)
IPDAF_tracker = tracking.IPDAFTracker(P_D, target_model, gate, P_Markov, gate.gamma)
IPDAInitiation = track_initiation.IPDAInitiation(initiate_thresh, terminate_thresh, IPDAF_tracker, gate)

initiation_type = 1   # 0: MofN     else: IPDA
if initiation_type == 0:
    track_termination = tracking.TrackTerminatorMofN(N_terminate)
    track_manager = tracking.Manager(PDAF_tracker, M_of_N, track_termination)
else:
    track_termination = tracking.TrackTerminatorIPDA(terminate_thresh)
    track_manager = tracking.Manager(IPDAF_tracker, IPDAInitiation, track_termination)

# Generate target trajectory - random turns, constant velocity
traj_tools = trajectory_tools.TrajectoryChange()
for k, t in enumerate(time):
    for ship in range(num_ships):
        if k == 0:
            x_true[ship, :, k] = x0[ship]
        else:
            x_true[ship, :, k] = F.dot(x_true[ship, :, k - 1])
            x_true[ship, :, k] = traj_tools.randomize_direction(x_true[ship, :, k]).reshape(4)


# Run tracking: Test scenario
measurements_all = []
for k, timestamp in enumerate(time):
    measurements = radar.generate_measurements([H.dot(x_true[ship, :, k]) for ship in range(num_ships)], timestamp)
    # measurements = radar.generate_clutter_measurements(timestamp)
    measurements_all.append(measurements)
    track_manager.step(measurements)
    if k % 10 == 0:
        print(k)

# ------------------------------------------------------------------------------

# Plot
fig, ax = visualization.plot_measurements(measurements_all)
# fig, ax = visualization.setup_plot(None)
for ship in range(num_ships):
    # ax.plot(x_true[ship, 2, 0:100], x_true[ship, 0, 0:100], 'k', label='True trajectory '+str(ship+1))
    ax.plot(x_true[ship, 2, :], x_true[ship, 0, :], 'k', label='True trajectory ' + str(ship + 1))
    ax.plot(x_true[ship, 2, 0], x_true[ship, 0, 0], 'ko')
visualization.plot_track_pos(track_manager.track_file, ax, 'r')
ax.set_xlim(-radar_range, radar_range)
ax.set_ylim(-radar_range, radar_range)
ax.set_xlabel('East[m]')
ax.set_ylabel('North[m]')
ax.set_title('Track position with sample rate: 1/s')
ax.legend()

# Analysis on completed test scenario
# analysis_sim.error_estimates(track_manager.track_file, x_true, t_end, c1, c2)
# analysis_sim.existence_confirmed_tracks(track_manager.track_file)
# analysis_sim.dual_plot_sim(measurements_all, num_ships, track_manager.track_file, x_true)

# Run analysis
# analysis_sim.roc(P_D, target_model, gate, P_Markov, initiate_thresh, terminate_thresh,
#         N_terminate, radar, c2, x_true, H, time)
# analysis_sim.existence(IPDAF_tracker, IPDAInitiation, track_termination, radar, x_true,
#                        H, num_ships, time)
# analysis_sim.false_tracks(P_D, target_model, gate, M_req, N_test, N_terminate, initiate_thresh, terminate_thresh,
#                  P_Markov, radar_range, R, time)
# analysis_sim.rmse(P_D, target_model, gate, initiate_thresh, terminate_thresh, P_Markov, time, x_true,
#                   H, num_ships, radar)
# analysis_sim.error_distances_plot(IPDAF_tracker, IPDAInitiation, track_termination, x_true, radar, time,
#                                   H, num_ships, t_end)
analysis_sim.true_tracks(PDAF_tracker, M_of_N, IPDAF_tracker, IPDAInitiation, N_terminate, terminate_thresh,
                         time, x_true, num_ships, H, radar, c2)

plt.show()
