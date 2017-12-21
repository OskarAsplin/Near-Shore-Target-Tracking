import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tracking
import simulation
import visualization
import track_initiation
import trajectory_tools

import pickle

# Load real data -------------------------------------------
def load_pkl(pkl_file):
    obj = None
    with open(pkl_file, 'rb') as f:
        obj = pickle.load(f,encoding='latin1')
    return obj

z_file = 'measurements_out.pkl'
# ais_file = 'ais_data_out.pkl'
ais_file = 'sync_ais_data_out.pkl'
measurements_all = load_pkl(z_file)
ais = load_pkl(ais_file)

measurements_all_new = []
for measurements in measurements_all:
    measurements_new = []
    for measurement in measurements:
        timestamp, value, covariance = measurement
        measurement_new = tracking.Measurement(value, timestamp, covariance)
        measurements_new.append(measurement_new)
    measurements_all_new.append(measurements_new)


for mmsi, data in ais.items():
    if mmsi != 257999459:
        time1, true_state1 = data
    else:
        time2, true_state2 = data

# Real data loaded -----------------------------------------

# Global constants
clutter_density = 2e-5
radar_range = 1000
dt = 2.88

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
initiate_thresh = 0.98
terminate_thresh = 0.10
# MofN
N_test = 6
M_req = 4
N_terminate = 3

# Set up tracking system
v_max = 10*dt
radar = simulation.SquareRadar(radar_range, clutter_density, P_D, R)
gate = tracking.TrackGate(P_G, v_max)
target_model = tracking.DWNAModel(q)

initiation_type = 0   # 0: MofN     else: IPDA
if initiation_type == 1:
    PDAF_tracker = tracking.PDAFTracker(P_D, target_model, gate)
    M_of_N = track_initiation.MOfNInitiation(M_req, N_test, PDAF_tracker, gate)
    track_termination = tracking.TrackTerminatorMofN(N_terminate)
    track_manager = tracking.Manager(PDAF_tracker, M_of_N, track_termination)
else:
    IPDAF_tracker = tracking.IPDAFTracker(P_D, target_model, gate, P_Markov, gate.gamma)
    IPDAInitiation = track_initiation.IPDAInitiation(initiate_thresh, terminate_thresh, IPDAF_tracker, gate)
    track_termination = tracking.TrackTerminatorIPDA(terminate_thresh)
    track_manager = tracking.Manager(IPDAF_tracker, IPDAInitiation, track_termination)



# Run tracking
for k, measurements in enumerate(measurements_all_new):
    track_manager.step(measurements)
    if k%10 == 0:
        print(k)

# vx_avg = -2.29914494171
# vy_avg = -1.16593075546
# t0 = time2[0]
# p0 = [true_state2[0, 0], true_state2[0, 2]]


# Run RMSE
errors_IPDA = dict()
c2 = 50


# Check if true tracks have been detected
k = 0
for track_id, state_list in track_manager.track_file.items():
    for est in state_list:
        t = est.timestamp
        # dt = t - t0
        idx, _ = trajectory_tools.find_nearest(time2, t)

        # est_target_pos = [p0[0] + vx_avg*dt, p0[1] + vy_avg*dt]
        # x1 = est_target_pos[0]
        # y1 = est_target_pos[1]
        x2 = est.est_posterior[0]
        y2 = est.est_posterior[2]
        dist = np.sqrt((true_state2[idx, 0]-x2)**2+(true_state2[idx, 2]-y2)**2)
        if dist < c2:
            errors_IPDA[k + 1] = dist
        k += 1


# for scan in errors_IPDA:
#     errors_IPDA[scan] = sum(errors_IPDA[scan]) / len(errors_IPDA[scan])

maxValue = max(errors_IPDA.values())
maxKey = max(errors_IPDA.keys())

list_IPDA = sorted(errors_IPDA.items())
xIPDA, yIPDA = zip(*list_IPDA)
print(sum(yIPDA[16:64])/len(yIPDA[16:64]))
# print(maxValue)
# print("scan numbers: ", xIPDA)
# print("Distances: ", yIPDA)

# Plot
fig, ax = visualization.setup_plot(None)
plt.plot(xIPDA, yIPDA, label='IPDA')
ax.set_title('Error distance')
ax.set_xlabel('Scan number')
ax.set_ylabel('Distance from real target [m]')
ax.legend()
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.ylim([0, maxValue])
plt.xlim([1, maxKey])
plt.show()