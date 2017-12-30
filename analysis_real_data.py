import numpy as np
import matplotlib.pyplot as plt
import visualization
import trajectory_tools
import matplotlib.ticker as ticker
import tracking
import track_initiation


def rmse(track_file, true_state, time):
    print('Running RMSE analysis')
    errors_IPDA = dict()
    c2 = 50

    # Check if true tracks have been detected
    k = 0
    for track_id, state_list in track_file.items():
        for est in state_list:
            t = est.timestamp
            idx, _ = trajectory_tools.find_nearest(time, t)
            dist = np.hypot(true_state[idx, 0] - est.est_posterior[0], true_state[idx, 2] - est.est_posterior[2])
            if dist < c2:
                errors_IPDA[k + 1] = dist
            k += 1

    # for scan in errors_IPDA:
    #     errors_IPDA[scan] = sum(errors_IPDA[scan]) / len(errors_IPDA[scan])

    maxValue = max(errors_IPDA.values())
    maxKey = max(errors_IPDA.keys())

    list_IPDA = sorted(errors_IPDA.items())
    xIPDA, yIPDA = zip(*list_IPDA)

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
    # plt.show()


def roc(P_D, target_model, gate, P_Markov, initiate_thresh, terminate_thresh,
        N_terminate, measurements_all_new, true_state, time):
    print('Running ROC analysis')
    true_IPDA = dict()
    false_IPDA = dict()
    true_MofN = dict()
    false_MofN = dict()

    # num_runs = 1
    c1 = 25
    c2 = 50
    true_IPDA_arr = []
    false_IPDA_arr = []
    true_MofN_arr = []
    false_MofN_arr = []
    init_values = [0.995, 0.98, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.51]
    # # time_int 10 & 12
    M_values = [8, 7, 6, 6, 5, 4, 4, 3, 3]
    N_values = [8, 7, 6, 7, 6, 5, 6, 5, 6]

    # # time_int 8
    # M_values = [6, 5, 4, 4, 3, 3, 2]
    # N_values = [6, 6, 5, 6, 5, 6, 6]

    # # time_int 6
    # M_values = [4, 3, 3, 2]
    # N_values = [4, 3, 4, 3]
    num_IPDA_tests = len(init_values)
    num_MofN_tests = len(M_values)
    len_time = 64
    time_int = 12
    num_runs = len_time - time_int
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

                for k, measurements in enumerate(measurements_all_new[run:run + time_int]):
                    track_manager.step(measurements)

                # Check if true tracks have been detected
                num_false = track_manager.conf_tracks_total
                spotted = 0
                for track_id, state_list in track_manager.track_file.items():
                    true_track = 1
                    for est in state_list:
                        t = est.timestamp
                        t_id, _ = trajectory_tools.find_nearest(time, t)
                        dist = np.hypot(true_state[t_id, 0] - est.est_posterior[0],
                                        true_state[t_id, 2] - est.est_posterior[2])
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
                    print("%.1f" % (100 * (run + para_test * num_runs + method * num_IPDA_tests * num_runs) /
                                    ((num_IPDA_tests + num_MofN_tests) * num_runs)), "% done")
            if method == 0:
                true_IPDA[initiate_thresh] = true_tracks / num_runs
                false_IPDA[initiate_thresh] = false_tracks / num_runs
                true_IPDA_arr.append(true_IPDA[initiate_thresh])
                false_IPDA_arr.append(false_IPDA[initiate_thresh])
            else:
                true_MofN[str(M_req) + " of " + str(N_test)] = true_tracks / num_runs
                false_MofN[str(M_req) + " of " + str(N_test)] = false_tracks / num_runs
                true_MofN_arr.append(true_MofN[str(M_req) + " of " + str(N_test)])
                false_MofN_arr.append(false_MofN[str(M_req) + " of " + str(N_test)])

    print("True IPDA: ", true_IPDA)
    print("False IPDA: ", false_IPDA)
    print("True MofN: ", true_MofN)
    print("False MofN: ", false_MofN)
    print("Arrays:")
    print("True IPDA: ", true_IPDA_arr)
    print("False IPDA: ", false_IPDA_arr)
    print("True MofN: ", true_MofN_arr)
    print("False MofN: ", false_MofN_arr)

    # Plot

    fig, ax = visualization.setup_plot(None)
    plt.plot(false_IPDA_arr, true_IPDA_arr, label='IPDA')
    plt.plot(false_MofN_arr, true_MofN_arr, label='M of N')

    # plt.semilogx(false_IPDA_arr, true_IPDA_arr, label='IPDA')
    # plt.semilogx(false_MofN_arr, true_MofN_arr, label='M of N')
    ax.set_title('ROC')
    ax.set_xlabel(r'$P_{FT}$')
    ax.set_ylabel(r'$P_{DT}$')
    ax.legend()
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.show()
