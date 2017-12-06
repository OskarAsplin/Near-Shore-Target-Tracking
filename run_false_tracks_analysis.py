import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import trajectory_tools
import tracking
import simulation
import visualization
import track_initiation


# Global constants
clutter_density = 1e-5
radar_range = 1000

# Initialized target
num_ships = 1
x0_1 = np.array([100, 4, 0, 5])
x0_2 = np.array([-100, -4, 0, -5])
x0 = [x0_1, x0_2]

# Time for simulation
dt = 1
t_end = 500
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


# Run true detected tracks demo
clutter_MofN = dict()
clutter_IPDA = dict()
num_scans = K
clut_arr = [4e-5, 3.5e-5, 3e-5, 2.5e-5, 2e-5, 1.5e-5, 1e-5, 5e-6]
for method in range(2):
    clut_it = -1
    clutter_density = clut_arr[clut_it]
    for run in range(len(clut_arr)):
        # Run tracking
        if method == 0:
            PDAF_tracker = tracking.PDAFTracker(P_D, target_model, gate)
            M_of_N = track_initiation.MOfNInitiation(M_req, N_test, PDAF_tracker, gate)
            track_termination = tracking.TrackTerminatorMofN(N_terminate)
            track_manager = tracking.Manager(PDAF_tracker, M_of_N, track_termination)
        else:
            IPDAF_tracker = tracking.IPDAFTracker(P_D, target_model, gate, P_Markov, gate.gamma)
            IPDAInitiation = track_initiation.IPDAInitiation(initiate_thresh, terminate_thresh, IPDAF_tracker, gate)
            track_termination = tracking.TrackTerminatorIPDA(terminate_thresh)
            track_manager = tracking.Manager(IPDAF_tracker, IPDAInitiation, track_termination)

        clut_it += 1
        clutter_density = clut_arr[clut_it]
        print(clutter_density)
        radar = simulation.SquareRadar(radar_range, clutter_density, P_D, R)
        for k, timestamp in enumerate(time):
            measurements = radar.generate_clutter_measurements(timestamp)
            track_manager.step(measurements)
            # Print time for debugging purposes
            if k%50==0:
                print(track_manager.conf_tracks_total)
        if method == 0:
            clutter_MofN[clutter_density] = track_manager.conf_tracks_total
        else:
            clutter_IPDA[clutter_density] = track_manager.conf_tracks_total


list_MofN = sorted(clutter_MofN.items())
list_IPDA = sorted(clutter_IPDA.items())
xMofN, yMofN = zip(*list_MofN)
xIPDA, yIPDA = zip(*list_IPDA)
print("Densities: ", xIPDA)
print("False tracks: ", yIPDA)

# Plot
fig, ax = visualization.setup_plot(None)
plt.plot(xMofN, yMofN, '--', label='M of N')
plt.plot(xIPDA, yIPDA, label='IPDA')
ax.set_title('False tracks detected over 500 scans')
ax.set_xlabel('Clutter density')
ax.set_ylabel('False tracks detected')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax.legend()
# for axis in [ax.xaxis, ax.yaxis]:
#     axis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.show()

