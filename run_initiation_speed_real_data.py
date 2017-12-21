import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import trajectory_tools
import tracking
import simulation
import visualization
import track_initiation
from matplotlib.ticker import LogLocator, AutoLocator

import pickle

# Load real data -------------------------------------------
def load_pkl(pkl_file):
    obj = None
    with open(pkl_file, 'rb') as f:
        obj = pickle.load(f,encoding='latin1')
    return obj

z_file = 'measurements_out.pkl'
ais_file = 'ais_data_out.pkl'
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
    time, true_state = data

# Real data loaded -----------------------------------------

est_true = np.array([[ -2.87403107e+02,-2.62901159e+00, 7.43773926e+02, 3.79407729e+00],
                     [ -2.94971741e+02,-2.62901159e+00, 7.54696655e+02, 3.79407729e+00],
                     [ -2.98589115e+02,-1.77116373e+00, 7.48783517e+02, 1.38434287e-01],
                     [ -3.04697770e+02,-1.95415548e+00, 7.44814922e+02,-5.81481834e-01],
                     [ -3.09878854e+02,-1.88607424e+00, 7.34182190e+02,-2.07886805e+00],
                     [ -3.12002032e+02,-1.30253474e+00, 7.27410523e+02,-2.21305109e+00],
                     [ -3.24488120e+02,-2.87083822e+00, 7.26018854e+02,-1.32230625e+00],
                     [ -3.52227923e+02,-4.66452797e+00, 7.41253977e+02, 2.51656886e+00],
                     [ -3.54834481e+02,-3.12169537e+00, 7.35270156e+02, 7.52378461e-01],
                     [ -3.65652415e+02,-3.39974404e+00, 7.40019445e+02, 1.13646237e+00],
                     [ -3.67919086e+02,-2.07976067e+00, 7.30840634e+02,-1.05499204e+00],
                     [ -3.55140179e+02, 1.21649437e+00, 6.97894417e+02,-4.30498995e+00],
                     [ -3.77449264e+02,-1.91600730e+00, 7.09937379e+02,-1.62008682e+00],
                     [ -3.89199108e+02,-2.58417864e+00, 7.08790760e+02,-1.36622520e+00],
                     [ -3.90487353e+02,-1.81068071e+00, 6.93107033e+02,-3.12194252e+00],
                     [ -3.97209737e+02,-2.06995475e+00, 6.84754476e+02,-3.02243386e+00],
                     [ -3.99117091e+02,-1.33278666e+00, 6.77211102e+02,-2.81593270e+00],
                     [ -4.00347067e+02,-8.63213163e-01, 6.71018359e+02,-2.47089154e+00],
                     [ -4.05037560e+02,-1.25300040e+00, 6.62887664e+02,-2.65455826e+00],
                     [ -4.15181172e+02,-2.22635030e+00, 6.70156233e+02,-4.01535146e-01],
                     [ -4.25263131e+02,-2.77913772e+00, 6.75423427e+02, 5.22755837e-01],
                     [ -4.34880232e+02,-3.07743259e+00, 6.73843005e+02,-6.49818612e-03],
                     [ -4.36199354e+02,-1.71461969e+00, 6.62838097e+02,-1.98868012e+00],
                     [ -4.51767748e+02,-3.27812088e+00, 6.62869823e+02,-1.27709639e+00],
                     [ -4.61204231e+02,-3.27812088e+00, 6.59193541e+02,-1.27709639e+00],
                     [ -4.64640753e+02,-2.43654255e+00, 6.62625991e+02,-2.95758250e-01],
                     [ -4.68912927e+02,-1.99424894e+00, 6.54822403e+02,-1.40577018e+00],
                     [ -4.74657563e+02,-1.99424894e+00, 6.50772939e+02,-1.40577018e+00],
                     [ -4.75252432e+02,-1.26071689e+00, 6.39655481e+02,-2.40549252e+00],
                     [ -4.86842710e+02,-2.54253462e+00, 6.34772992e+02,-2.08913041e+00],
                     [ -4.95214027e+02,-2.72273829e+00, 6.31137241e+02,-1.66988049e+00],
                     [ -5.01579961e+02,-2.45709447e+00, 6.26210814e+02,-1.69101988e+00],
                     [ -5.08657649e+02,-2.45709447e+00, 6.21339813e+02,-1.69101988e+00],
                     [ -5.11971614e+02,-1.91711477e+00, 6.17464150e+02,-1.54824721e+00],
                     [ -5.18428525e+02,-2.06906819e+00, 6.13590531e+02,-1.45323998e+00],
                     [ -5.24379638e+02,-2.06906819e+00, 6.09410681e+02,-1.45323998e+00],
                     [ -5.35813724e+02,-2.85678882e+00, 6.04237124e+02,-1.59573036e+00],
                     [ -5.38570762e+02,-1.97074041e+00, 6.03209399e+02,-1.01509615e+00],
                     [ -5.42167668e+02,-1.60097950e+00, 5.98254482e+02,-1.37869890e+00],
                     [ -5.51192861e+02,-2.40138604e+00, 5.93592241e+02,-1.50396600e+00],
                     [ -5.55965233e+02,-2.01560177e+00, 5.91879930e+02,-1.03045128e+00],
                     [ -5.73044332e+02,-4.04723811e+00, 5.87329999e+02,-1.31410735e+00],
                     [ -5.79776015e+02,-3.40016819e+00, 5.86471172e+02,-8.77037182e-01],
                     [ -5.88049469e+02,-3.16656972e+00, 5.82548653e+02,-1.11758952e+00],
                     [ -5.91173868e+02,-2.10104683e+00, 5.77922049e+02,-1.37281091e+00],
                     [ -5.98644865e+02,-2.35564276e+00, 5.76860919e+02,-8.50698873e-01],
                     [ -6.10154136e+02,-3.21136775e+00, 5.68387239e+02,-1.93889787e+00],
                     [ -6.17208136e+02,-2.81905329e+00, 5.61272847e+02,-2.21370955e+00],
                     [ -6.24642998e+02,-2.69844126e+00, 5.55946428e+02,-2.02726228e+00],
                     [ -6.24402240e+02,-1.25172731e+00, 5.55825358e+02,-9.94877691e-01],
                     [ -6.30587434e+02,-1.71354926e+00, 5.57363908e+02,-2.12867855e-01],
                     [ -6.39976660e+02,-2.50175223e+00, 5.55706448e+02,-3.94598306e-01],
                     [ -6.49056150e+02,-2.83721502e+00, 5.47683338e+02,-1.63225139e+00],
                     [ -6.55850303e+02,-2.59359081e+00, 5.46627310e+02,-9.88263976e-01],
                     [ -6.64425529e+02,-2.79201115e+00, 5.42169285e+02,-1.27576853e+00],
                     [ -6.72554360e+02,-2.80748365e+00, 5.37491177e+02,-1.45667139e+00],
                     [ -6.79004761e+02,-2.51378339e+00, 5.32310295e+02,-1.63598013e+00],
                     [ -6.87302010e+02,-2.70507681e+00, 5.27191786e+02,-1.70973544e+00],
                     [ -6.89737041e+02,-1.73676623e+00, 5.24481093e+02,-1.30925481e+00],
                     [ -6.94921832e+02,-1.76936439e+00, 5.20695333e+02,-1.31181081e+00],
                     [ -7.05094520e+02,-2.68481205e+00, 5.14426090e+02,-1.76172806e+00],
                     [ -7.12172834e+02,-2.56795075e+00, 5.11442098e+02,-1.38555906e+00],
                     [ -7.19897660e+02,-2.62891125e+00, 5.08064019e+02,-1.27592896e+00],
                     [ -7.25815241e+02,-2.33069495e+00, 5.00263234e+02,-2.02203803e+00]])

est_time_true_states = np.array([1507795069.24, 1507795072.12, 1507795075.00, 1507795077.88,
                                1507795080.76, 1507795083.64, 1507795086.52, 1507795092.27,
                                1507795095.15, 1507795098.03, 1507795100.91, 1507795106.67,
                                1507795109.55, 1507795112.43, 1507795115.31, 1507795118.19,
                                1507795121.07, 1507795123.95, 1507795126.82, 1507795129.70,
                                1507795132.58, 1507795135.46, 1507795138.34, 1507795141.22,
                                1507795144.09, 1507795146.98, 1507795149.85, 1507795152.73,
                                1507795155.61, 1507795158.49, 1507795161.37, 1507795164.25,
                                1507795167.13, 1507795170.01, 1507795172.89, 1507795175.76,
                                1507795178.64, 1507795181.52, 1507795184.40, 1507795187.28,
                                1507795190.16, 1507795193.03, 1507795195.92, 1507795198.79,
                                1507795201.67, 1507795204.55, 1507795207.43, 1507795210.31,
                                1507795213.19, 1507795216.07, 1507795218.95, 1507795221.83,
                                1507795224.71, 1507795227.58, 1507795230.46, 1507795233.34,
                                1507795236.22, 1507795239.10, 1507795241.98, 1507795244.86,
                                1507795247.73, 1507795250.62, 1507795253.49, 1507795256.37])


# Global constants
clutter_density = 2e-5
radar_range = 1000
dt = 2

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


# Run true detected tracks demo
num_IPDA = dict()
num_MofN = dict()

num_runs = 1
c1 = 25
c2 = 50
num_IPDA_arr = []
num_MofN_arr = []
init_values = [0.995, 0.98, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.51]
M_values = [8, 7, 6, 6, 5, 4, 4, 3, 3]
N_values = [8, 7, 6, 7, 6, 5, 6, 5, 6]
#M_values = [6, 5, 4, 4, 3, 3, 2]
#N_values = [6, 6, 5, 6, 5, 6, 6]
num_IPDA_tests = len(init_values)
num_MofN_tests = len(M_values)

initiate_thresh = 0.505

for method in range(2):
    init_it = -1
    if method == 0:
        continue
    for para_test in range(num_IPDA_tests if method == 0 else num_MofN_tests):
    # while True:
        init_it += 1
        if method == 0:
            #initiate_thresh = init_values[init_it]
            initiate_thresh = initiate_thresh + 0.01
            if initiate_thresh >= 1:
                break
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

            track_spotted = 0
            for k, measurements in enumerate(measurements_all_new):
                track_manager.step(measurements)

                # Check if true tracks have been detected
                for track_id, state_list in track_manager.track_file.items():
                    for est in state_list:
                        t = est.timestamp
                        t_id, _ = trajectory_tools.find_nearest(est_time_true_states, t)
                        dist = np.sqrt((est_true[t_id, 2] - est.est_posterior[2]) ** 2 + (est_true[t_id, 0] - est.est_posterior[0]) ** 2)
                        if dist < c2:
                            if method == 0:
                                num_IPDA[initiate_thresh] = k + 1
                            else:
                                num_MofN[str(M_req)+" of "+str(N_test)] = k + 1
                            track_spotted = 1
                            break

                if track_spotted == 1:
                    break


        if method == 0:
            num_IPDA_arr.append(num_IPDA[initiate_thresh])
        else:
            num_MofN_arr.append(num_MofN[str(M_req)+" of "+str(N_test)])


print("True IPDA: ", num_IPDA)
print("True MofN: ", num_MofN)
print("Arrays:")
print("True IPDA: ", num_IPDA_arr)
print("True MofN: ", num_MofN_arr)



fig, ax = visualization.setup_plot(None)
plt.plot(list(num_IPDA.keys()), num_IPDA_arr, label='IPDA')
#plt.plot(list(num_MofN.keys()), num_MofN_arr, label='M of N')

#plt.semilogx(num_IPDA_arr, true_IPDA_arr, '-', label='IPDA')
#plt.semilogx(num_MofN_arr, true_MofN_arr, '-', label='M of N')
ax.set_title('Number of scans needed to initiate true target for IPDA')
ax.set_xlabel('Scans')
ax.set_ylabel('Initiate threshold')
ax.legend()
# for axis in [ax.xaxis, ax.yaxis]:
#     axis.set_major_locator(ticker.MaxNLocator(integer=True))
#plt.ylim([0, 1])
#plt.xlim([0, 1])
plt.show()