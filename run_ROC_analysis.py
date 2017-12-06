import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import trajectory_tools
import tracking
import simulation
import visualization
import track_initiation


# Global constants
clutter_density = 2e-5
radar_range = 1000

# Initialized target
num_ships = 1
x0_1 = np.array([100, 4, 0, 5])
x0_2 = np.array([-100, -4, 0, -5])
x0 = [x0_1, x0_2]

# Time for simulation
dt = 1
t_end = 10
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
initiate_thresh = 0.90
terminate_thresh = 0.10
# MofN
N_test = 6
M_req = 4
N_terminate = 3

# Set up tracking system
v_max = 10/dt
radar = simulation.SquareRadar(radar_range, clutter_density, P_D, R)
gate = tracking.TrackGate(P_G, v_max)
target_model = tracking.DWNAModel(q)

# Generate target trajectory - random turns, constant velocity
traj_tools = trajectory_tools.TrajectoryChange()
for k, t in enumerate(time):
    for ship in range(num_ships):
        if k == 0:
            x_true[ship, :, k] = x0[ship]
        else:
            x_true[ship, :, k] = F.dot(x_true[ship, :, k - 1])
            x_true[ship, :, k] = traj_tools.randomize_direction(x_true[ship, :, k]).reshape(4)

# Run true detected tracks demo
true_IPDA = dict()
false_IPDA = dict()
true_MofN = dict()
false_MofN = dict()

num_runs = 2000
c1 = 25
c2 = 50
true_IPDA_arr = []
false_IPDA_arr = []
true_MofN_arr = []
false_MofN_arr = []
init_values = [0.995, 0.98, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.51]
M_values = [8, 7, 6, 6, 5, 4, 4, 3, 3]
N_values = [8, 7, 6, 7, 6, 5, 6, 5, 6]
num_IPDA_tests = len(init_values)
num_MofN_tests = len(M_values)
for method in range(2):
    init_it = -1
    for para_test in range(num_IPDA_tests if method == 0 else num_MofN_tests):
        init_it += 1
        if method == 0:
            initiate_thresh = init_values[init_it]
        else:
            M_req = M_values[init_it]
            N_test = N_values[init_it]
        true_tracks = 0
        false_tracks = 0
        for run in range(num_runs):
            # Run tracking
            if method == 0:
                IPDAF_tracker = tracking.IPDAFTracker(P_D, target_model, gate, P_Markov, gate.gamma)
                IPDAInitiation = track_initiation.IPDAInitiation(initiate_thresh, terminate_thresh, IPDAF_tracker, gate)
                track_termination = tracking.TrackTerminatorIPDA(terminate_thresh)
                track_manager = tracking.Manager(IPDAF_tracker, IPDAInitiation, track_termination)
            else:
                PDAF_tracker = tracking.PDAFTracker(P_D, target_model, gate)
                M_of_N = track_initiation.MOfNInitiation(M_req, N_test, PDAF_tracker, gate)
                track_termination = tracking.TrackTerminatorMofN(N_terminate)
                track_manager = tracking.Manager(PDAF_tracker, M_of_N, track_termination)

            for k, timestamp in enumerate(time):
                measurements = radar.generate_measurements([H.dot(x_true[ship, :, k]) for ship in range(num_ships)], timestamp)
                track_manager.step(measurements)

            # Check if true tracks have been detected
            num_false = track_manager.conf_tracks_total
            spotted = 0
            for track_id, state_list in track_manager.track_file.items():
                true_track = 1
                for est in state_list:
                    t = est.timestamp
                    dist = trajectory_tools.dist(x_true[0, 2, t], x_true[0, 0, t], est.est_posterior[2], est.est_posterior[0])
                    if dist > c2:
                        true_track = 0
                        break
                if true_track == 1:
                    num_false -= 1
                    spotted = 1
            false_tracks += min(num_false, 1)
            true_tracks += spotted

            # Print run number for debugging purposes
            if run % 100 == 0:
                print("%.1f" % (100 * (run+para_test*num_runs+method*num_IPDA_tests*num_runs) /
                                ((num_IPDA_tests+num_MofN_tests) * num_runs)), "% done")
        if method == 0:
            true_IPDA[initiate_thresh] = true_tracks / num_runs
            false_IPDA[initiate_thresh] = false_tracks / num_runs
            true_IPDA_arr.append(true_IPDA[initiate_thresh])
            false_IPDA_arr.append(false_IPDA[initiate_thresh])
        else:
            true_MofN[str(M_req)+" of "+str(N_test)] = true_tracks / num_runs
            false_MofN[str(M_req)+" of "+str(N_test)] = false_tracks / num_runs
            true_MofN_arr.append(true_MofN[str(M_req)+" of "+str(N_test)])
            false_MofN_arr.append(false_MofN[str(M_req)+" of "+str(N_test)])

print("True IPDA: ", true_IPDA)
print("False IPDA: ", false_IPDA)
print("True MofN: ", true_MofN)
print("False MofN: ", false_MofN)
print("Arrays:")
print("True IPDA: ", true_IPDA_arr)
print("False IPDA: ", false_IPDA_arr)
print("True MofN: ", true_MofN_arr)
print("False MofN: ", false_MofN_arr)

# list_IPDA = sorted(true_IPDA.items())
# xIPDA, yIPDA = zip(*list_IPDA)

# Plot
fig, ax = visualization.setup_plot(None)
plt.plot(false_IPDA_arr, true_IPDA_arr, label='IPDA')
plt.plot(false_MofN_arr, true_MofN_arr, label='M of N')
ax.set_title('ROC')
ax.set_xlabel(r'$P_{FA}$')
ax.set_ylabel(r'$P_D$')
ax.legend()
# for axis in [ax.xaxis, ax.yaxis]:
#     axis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.show()