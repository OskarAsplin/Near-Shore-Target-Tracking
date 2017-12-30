import numpy as np
import matplotlib.pyplot as plt
import visualization
import matplotlib.ticker as ticker
import tracking
import track_initiation
import simulation


def true_tracks(PDAF_tracker, M_of_N, IPDAF_tracker, IPDAInitiation, N_terminate, terminate_thresh, time, x_true,
                num_ships, H, radar, c2):
    print('Starting true tracks analysis')
    scans_MofN = dict()
    scans_IPDA = dict()
    num_runs = 500
    for method in range(2):
        for run in range(num_runs):
            # Run tracking
            if method == 0:
                track_termination = tracking.TrackTerminatorMofN(N_terminate)
                track_manager = tracking.Manager(PDAF_tracker, M_of_N, track_termination)
            else:
                track_termination = tracking.TrackTerminatorIPDA(terminate_thresh)
                track_manager = tracking.Manager(IPDAF_tracker, IPDAInitiation, track_termination)

            tracks_spotted = set()
            for k, timestamp in enumerate(time):
                measurements = radar.generate_measurements([H.dot(x_true[ship, :, k]) for ship in range(num_ships)],
                                                           timestamp)
                track_manager.step(measurements)

                # Check if true tracks have been detected
                for track_id, state_list in track_manager.track_file.items():
                    states = np.array([est.est_posterior for est in state_list])
                    for ship in range(num_ships):
                        if np.hypot(x_true[ship, 0, k] - states[-1, 0], x_true[ship, 2, k] - states[-1, 2]) < c2:
                            tracks_spotted.add(ship)
                            break
                if len(tracks_spotted) == num_ships:
                    if method == 0:
                        if k + 1 in scans_MofN:
                            scans_MofN[k + 1] += 1
                        else:
                            scans_MofN[k + 1] = 1
                    else:
                        if k + 1 in scans_IPDA:
                            scans_IPDA[k + 1] += 1
                        else:
                            scans_IPDA[k + 1] = 1
                    break

            # Print time for debugging purposes
            if run % 50 == 0:
                print(run)

    max_key = max(max(scans_MofN.keys()), max(scans_IPDA.keys()))

    for scans in [scans_MofN, scans_IPDA]:
        for key in range(1, max_key + 1):
            if key not in scans:
                scans[key] = 0

        last = 0
        for key in sorted(scans.keys()):
            last = last + scans[key]
            scans[key] = last

    list_MofN = sorted(scans_MofN.items())
    list_IPDA = sorted(scans_IPDA.items())
    xMofN, yMofN = zip(*list_MofN)
    xIPDA, yIPDA = zip(*list_IPDA)

    # Plot
    fig, ax = visualization.setup_plot(None)
    plt.plot(xMofN, yMofN, '--', label='M of N')
    plt.plot(xIPDA, yIPDA, label='IPDA')
    ax.set_title('True detected tracks out of 500')
    ax.set_xlabel('Scans needed')
    ax.set_ylabel('Detected tracks')
    ax.legend()
    # ax.grid()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    # plt.xlim([1, 20])
    plt.ylim([0, 500])


def error_distances_plot(IPDAF_tracker, IPDAInitiation, track_termination, x_true, radar, time, H, num_ships, t_end):
    print('Starting error distances plot')
    num_runs = 500
    error_arr = []
    for run in range(num_runs):
        track_manager = tracking.Manager(IPDAF_tracker, IPDAInitiation, track_termination)
        # Run tracking
        for k, timestamp in enumerate(time):
            measurements = radar.generate_measurements([H.dot(x_true[ship, :, k]) for ship in range(num_ships)],
                                                       timestamp)
            track_manager.step(measurements)

        # Error for estimates (One ship)
        for track_id, state_list in track_manager.track_file.items():
            error_dic = dict()
            for est in state_list:
                t = est.timestamp
                dist = np.hypot(x_true[0, 0, t] - est.est_posterior[0], x_true[0, 2, t] - est.est_posterior[2])
                error_dic[t] = dist
            error_arr.append(error_dic)
        if run % 10 == 0:
            print(run)

    # Plot
    fig, ax = visualization.setup_plot(None)
    for dic in error_arr:
        list_IPDA = sorted(dic.items())
        xIPDA, yIPDA = zip(*list_IPDA)
        plt.plot(xIPDA, yIPDA)
    ax.set_title('Error distance of 500 runs of 30 scans')
    ax.set_xlabel('Scan number')
    ax.set_ylabel('Distance from real target [m]')
    # ax.legend()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    c1 = 25
    c2 = 50
    plt.plot((0, t_end), (c1, c1), 'k--')
    plt.plot((0, t_end), (c2, c2), 'k--')


def rmse(P_D, target_model, gate, initiate_thresh, terminate_thresh, P_Markov, time, x_true, H, num_ships, radar, c2):
    print('Starting RMSE analysis')
    errors_IPDA = dict()
    num_runs = 100
    for run in range(num_runs):
        # Run tracking
        IPDAF_tracker = tracking.IPDAFTracker(P_D, target_model, gate, P_Markov, gate.gamma)
        IPDAInitiation = track_initiation.IPDAInitiation(initiate_thresh, terminate_thresh, IPDAF_tracker, gate)
        track_termination = tracking.TrackTerminatorIPDA(terminate_thresh)
        track_manager = tracking.Manager(IPDAF_tracker, IPDAInitiation, track_termination)

        for k, timestamp in enumerate(time):
            measurements = radar.generate_measurements([H.dot(x_true[ship, :, k]) for ship in range(num_ships)],
                                                       timestamp)
            track_manager.step(measurements)

            # Check if true tracks have been detected
            for track_id, state_list in track_manager.track_file.items():
                states = np.array([est.est_posterior for est in state_list])
                for ship in range(num_ships):
                    dist = np.hypot(x_true[ship, 0, k] - states[-1, 0], x_true[ship, 2, k] - states[-1, 2])
                    if dist < c2:
                        if k + 1 in errors_IPDA:
                            errors_IPDA[k + 1].append(dist)
                        else:
                            errors_IPDA[k + 1] = [dist]

        # Print time for debugging purposes
        if run % 50 == 0:
            print("%.1f" % (100 * run / num_runs), "% done")

    for scan in errors_IPDA:
        errors_IPDA[scan] = sum(errors_IPDA[scan]) / len(errors_IPDA[scan])

    maxValue = max(errors_IPDA.values())
    maxKey = max(errors_IPDA.keys())

    list_IPDA = sorted(errors_IPDA.items())
    xIPDA, yIPDA = zip(*list_IPDA)
    print("scan numbers: ", xIPDA)
    print("Distances: ", yIPDA)

    # Plot
    fig, ax = visualization.setup_plot(None)
    plt.plot(xIPDA, yIPDA, label='IPDA')
    ax.set_title('RMSE of 10 000 runs of 30 scans')
    ax.set_xlabel('Scan number')
    ax.set_ylabel('Distance from real target [m]')
    ax.legend()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.ylim([0, maxValue])
    plt.xlim([1, maxKey])


def false_tracks(P_D, target_model, gate, M_req, N_test, N_terminate, initiate_thresh, terminate_thresh,
                 P_Markov, radar_range, R, time):
    print('Starting false tracks analysis')
    clutter_MofN = dict()
    clutter_IPDA = dict()
    clut_arr = [4e-5, 3.5e-5, 3e-5, 2.5e-5, 2e-5, 1.5e-5, 1e-5, 5e-6]
    for method in range(2):
        clut_it = -1
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
                if k % 50 == 0:
                    print(track_manager.conf_tracks_total)
            if method == 0:
                clutter_MofN[clutter_density] = track_manager.conf_tracks_total
            else:
                clutter_IPDA[clutter_density] = track_manager.conf_tracks_total

    list_MofN = sorted(clutter_MofN.items())
    list_IPDA = sorted(clutter_IPDA.items())
    xMofN, yMofN = zip(*list_MofN)
    xIPDA, yIPDA = zip(*list_IPDA)
    print("Densities IPDA: ", xIPDA)
    print("False tracks IPDA: ", yIPDA)
    print("Densities M of N: ", xMofN)
    print("False tracks M of N: ", yMofN)

    # Plot
    fig, ax = visualization.setup_plot(None)
    plt.semilogy(xMofN, yMofN, '--', label='M of N')
    plt.semilogy(xIPDA, yIPDA, label='IPDA')
    ax.set_title('False tracks detected over 1000 scans')
    ax.set_xlabel('Clutter density')
    ax.set_ylabel('False tracks detected')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.legend()


def existence(IPDAF_tracker, IPDAInitiation, track_termination, radar, x_true, H, num_ships, time):
    print('Starting existence analysis')
    num_runs = 50
    exist_arr = []
    for run in range(num_runs):
        track_manager = tracking.Manager(IPDAF_tracker, IPDAInitiation, track_termination)
        for k, timestamp in enumerate(time):
            measurements = radar.generate_measurements([H.dot(x_true[ship, :, k]) for ship in range(num_ships)], timestamp)
            track_manager.step(measurements)

        # Existence
        for track_id, state_list in track_manager.track_file.items():
            exist_dic = dict()
            for est in state_list:
                t = est.timestamp
                exist_dic[t] = est.exist_posterior
            exist_arr.append(exist_dic)

    # Plot
    fig, ax = visualization.setup_plot(None)
    for dic in exist_arr:
        list_IPDA = sorted(dic.items())
        xIPDA, yIPDA = zip(*list_IPDA)
        plt.plot(xIPDA, yIPDA)
    ax.set_title('Existence for confirmed tracks')
    ax.set_xlabel('Scan number')
    ax.set_ylabel('Probability')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))


def dual_plot_sim(measurements_all, num_ships, track_file, x_true):
    print('Starting dual_plot_sim')
    f, (ax2, ax1) = plt.subplots(1, 2)
    fig, ax1 = visualization.plot_measurements(measurements_all, ax1)
    # fig, ax = visualization.setup_plot(None)
    for ship in range(num_ships):
        # ax.plot(x_true[ship, 2, 0:100], x_true[ship, 0, 0:100], 'k', label='True trajectory '+str(ship+1))
        ax1.plot(x_true[ship, 2, :], x_true[ship, 0, :], 'k', label='True trajectory ' + str(ship + 1))
        ax1.plot(x_true[ship, 2, 0], x_true[ship, 0, 0], 'ko')

    visualization.plot_track_pos(track_file, ax1, 'r')
    ax1.set_xlim(-250, 250)
    ax1.set_ylim(-250, 250)
    ax1.set_xlabel('East[m]')
    ax1.set_ylabel('North[m]')
    ax1.set_title('Track position with sample rate: 1/s')
    ax1.legend(loc="upper left")

    fig, ax2 = visualization.plot_measurements(measurements_all, ax2)
    # fig, ax = visualization.setup_plot(None)
    for ship in range(num_ships):
        # ax.plot(x_true[ship, 2, 0:100], x_true[ship, 0, 0:100], 'k', label='True trajectory '+str(ship+1))
        ax2.plot(x_true[ship, 2, :], x_true[ship, 0, :], 'k', label='True trajectory ' + str(ship + 1))
        ax2.plot(x_true[ship, 2, 0], x_true[ship, 0, 0], 'ko')
    # visualization.plot_track_pos(track_manager.track_file, ax, 'r')
    ax2.set_xlim(-250, 250)
    ax2.set_ylim(-250, 250)
    ax2.set_xlabel('East[m]')
    ax2.set_ylabel('North[m]')
    ax2.set_title('Track position with sample rate: 1/s')
    ax2.legend(loc="upper left")


def existence_confirmed_tracks(track_file):
    print('Starting existence of confirmed tracks')
    exist_arr = []
    for track_id, state_list in track_file.items():
        exist_dic = dict()
        for est in state_list:
            t = est.timestamp
            exist_dic[t] = est.exist_posterior
        exist_arr.append(exist_dic)

    # Plot
    fig, ax = visualization.setup_plot(None)
    for dic in exist_arr:
        list_IPDA = sorted(dic.items())
        xIPDA, yIPDA = zip(*list_IPDA)
        plt.plot(xIPDA, yIPDA)
    ax.set_title('Existence for confirmed tracks')
    ax.set_xlabel('Scan number')
    ax.set_ylabel('Probability')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))


def error_estimates(track_file, x_true, t_end, c1, c2):
    print('Starting error estimates (one ship)')
    error_arr = []
    for track_id, state_list in track_file.items():
        error_dic = dict()
        for est in state_list:
            t = est.timestamp
            dist = np.hypot(x_true[0, 0, t] - est.est_posterior[0], x_true[0, 2, t] - est.est_posterior[2])
            error_dic[t] = dist
        error_arr.append(error_dic)

    # Plot
    fig, ax = visualization.setup_plot(None)
    for dic in error_arr:
        list_IPDA = sorted(dic.items())
        xIPDA, yIPDA = zip(*list_IPDA)
        plt.plot(xIPDA, yIPDA)
    ax.set_title('RMSE of 200 runs of 30 scans')
    ax.set_xlabel('Scan number')
    ax.set_ylabel('Distance from real target [m]')
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.plot((0, t_end), (c1, c1), 'k--')
    plt.plot((0, t_end), (c2, c2), 'k--')
    # plt.ylim([0, maxValue])
    # plt.xlim([1, maxKey])
    # plt.show()


def roc(P_D, target_model, gate, P_Markov, initiate_thresh, terminate_thresh,
        N_terminate, radar, c2, x_true, H, time):
    print('Starting ROC analysis')
    true_IPDA = dict()
    false_IPDA = dict()
    true_MofN = dict()
    false_MofN = dict()

    num_runs = 2000
    true_IPDA_arr = []
    false_IPDA_arr = []
    true_MofN_arr = []
    false_MofN_arr = []
    init_values = [0.995, 0.98, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.51]
    # M_values = [8, 7, 6, 6, 5, 4, 4, 3, 3]
    # N_values = [8, 7, 6, 7, 6, 5, 6, 5, 6]
    # M_values = [6, 5, 4, 4, 3, 3]
    # N_values = [6, 6, 5, 6, 5, 6]
    M_values = [4, 3, 3, 2, 2]
    N_values = [4, 3, 4, 2, 3]
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
                    measurements = radar.generate_measurements([H.dot(x_true[0, :, k])], timestamp)
                    track_manager.step(measurements)

                # Check if true tracks have been detected
                num_false = track_manager.conf_tracks_total
                spotted = 0
                for track_id, state_list in track_manager.track_file.items():
                    true_track = 1
                    for est in state_list:
                        t = est.timestamp
                        dist = np.hypot(x_true[0, 0, t] - est.est_posterior[0], x_true[0, 2, t] - est.est_posterior[2])
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
    ax.set_title('ROC')
    ax.set_xlabel(r'$P_{FA}$')
    ax.set_ylabel(r'$P_D$')
    ax.legend()
    plt.ylim([0, 1])
    plt.xlim([0, 1])
