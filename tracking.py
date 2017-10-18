import numpy as np
from scipy.stats import chi2


class DefaultLogger(object):
    def __init__(self):
        pass

    def print_log(self, logstring):
        print(logstring)

    def logerr(self, log):
        self.print_log("Error: " + log)

    def loginfo(self, log):
        self.print_log("Info: " + log)

    def logdebug(self, log):
        self.print_log("Debug: " + log)


class Measurement(object):
    def __init__(self, value, timestamp, covariance, logger=DefaultLogger()):
        self.value = value
        self.timestamp = timestamp
        self.measurement_mapping = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
        self.covariance = covariance
        self.logger = logger

    def __repr__(self):
        meas_str = "Measurement: (%.2f, %.2f)" % (self.value[0], self.value[1])
        time_str = "Timestamp: %.2f" % (self.timestamp)
        return meas_str+", "+time_str

    def __getitem__(self, key):
        return self.value[key]


class Estimate(object):
    def __init__(self, t, mean, covariance, is_posterior=False, track_index=None):
        self.timestamp = t
        self.measurements = []
        self.est_prior = mean
        self.cov_prior = covariance
        if is_posterior:
            self.est_posterior = mean
            self.cov_posterior = covariance
        if track_index is not None:
            self.track_index = track_index
        else:
            self.track_index = -1

    def __repr__(self):
        ID_str = "Track ID: %d" % (self.track_index)
        timestamp_str = "Timestamp: %.2f" % self.timestamp
        return ID_str+ ", " + timestamp_str

    def store_measurement(self, measurement):
        self.measurements.append(measurement)


class DWNAModel(object):
    def __init__(self, q):
        self.q = q
        self.get_model = lambda t: self.model(t, self.q)

    def __repr__(self):
        return "DWNA model with variance q=%.2f m^2/s^4" % self.q

    def step(self, estimate, timestamp):
        dt = timestamp - estimate.timestamp
        F, Q = self.model(dt, self.q)
        est_new = F.dot(estimate.est_posterior)
        cov_new = F.dot(estimate.cov_posterior).dot(F.T) + Q
        new_estimate = Estimate(timestamp, est_new, cov_new, track_index=estimate.track_index)
        return new_estimate

    @staticmethod
    def model(t, q):
        F = np.identity(4)
        F[0, 1] = t
        F[2, 3] = t
        G = np.zeros((4, 2))
        G[0, 0] = t ** 2 / 2.0
        G[1, 0] = t
        G[2, 1] = t ** 2 / 2.0
        G[3, 1] = t
        Q = q * np.dot(G, G.T)
        return F, Q


class TrackGate(object):
    def __init__(self, gate_probability, v_max):
        self.gate_probability = gate_probability
        self.gamma = chi2(df=2).ppf(gate_probability)
        self.v_max = v_max

    def __repr__(self):
        return "Track gate with gate probability = %.2f, max. velocity %.1f m/2" % (self.gate_probability, self.v_max)

    def gate_estimate(self, estimate, measurements):
        measurements_used = set()
        for z_idx, measurement in enumerate(measurements):
            z = measurement.value
            H = measurement.measurement_mapping
            R = measurement.covariance
            estimate.z_hat = H.dot(estimate.est_prior)
            estimate.S = H.dot(estimate.cov_prior).dot(H.T)+R
            v_ik = z - estimate.z_hat
            if v_ik.T.dot(np.linalg.inv(estimate.S)).dot(v_ik) < self.gamma:
                estimate.store_measurement(measurement)
                measurements_used.add(z_idx)
        return measurements_used

    def gate_measurement(self, center_measurement, test_measurement):
        dt = test_measurement.timestamp - center_measurement.timestamp
        d_plus = np.maximum(test_measurement.value-center_measurement.value-dt*self.v_max, np.zeros(2))
        d_minus = np.maximum(-(test_measurement.value-center_measurement.value+dt*self.v_max), np.zeros(2))
        d = d_plus+d_minus
        R_initiator = center_measurement.covariance
        R_measurement = test_measurement.covariance
        D = d.dot(np.linalg.inv(R_initiator+R_measurement)).dot(d)
        return D < self.gamma

    def filter_measurements(self, center, measurements):
        measurements_inside = []
        for measurement in measurements:
            if self.gate_measurement(center, measurement):
                measurements_inside.append(measurement)
        return measurements_inside


class PDAFTracker(object):
    def __init__(self, P_D, target_model, gate_method):
        self.detection_probability = P_D
        self.target_model = target_model
        self.gate_method = gate_method

    def step(self, old_estimates, measurements, timestamp):
        estimates = [self.target_model.step(old_est, timestamp) for old_est in old_estimates]
        used_measurements = set()
        for estimate in estimates:
            new_used_measurements = self.gate_method.gate_estimate(estimate, measurements)
            used_measurements = used_measurements | new_used_measurements
            self.update_estimate(estimate)
        unused_measurements = [measurement for idx, measurement in enumerate(measurements) if idx not in used_measurements]
        return estimates, unused_measurements

    def update_estimate(self, estimate):
        n_measurements = len(estimate.measurements)
        P_D = self.detection_probability
        P_G = self.gate_method.gate_probability
        if n_measurements == 0:
            estimate.est_posterior = estimate.est_prior
            estimate.cov_posterior = estimate.cov_prior
            return
        H = estimate.measurements[0].measurement_mapping
        z_all = np.array([measurement.value for measurement in estimate.measurements]).T
        b = 2/self.gate_method.gamma*(1-P_D*P_G)/P_D*n_measurements
        e = np.zeros(n_measurements)
        innovations = np.zeros_like(z_all)
        for i in range(n_measurements):
            innovations[:,i] = z_all[:,i] - estimate.z_hat
            e[i] = np.exp(-0.5*innovations[:,i].dot(np.linalg.inv(estimate.S)).dot(innovations[:,i]))
        betas = np.hstack((e, b))
        betas = betas/(1.*np.sum(betas))
        kalman_gain = estimate.cov_prior.dot(H.T).dot(np.linalg.inv(estimate.S))
        total_innovation = np.zeros(2)
        cov_terms = np.zeros((2,2))
        for i in range(n_measurements):
            innov = innovations[:,i]
            total_innovation += betas[i]*innov
            innov_vec = innov.reshape((2,1))
            cov_terms += betas[i]*np.dot(innov_vec, innov_vec.T)
        estimate.est_posterior = estimate.est_prior+np.dot(kalman_gain, total_innovation)
        total_innovation_vec = total_innovation.reshape((2,1))
        cov_terms = cov_terms-np.dot(total_innovation_vec, total_innovation_vec.T)
        soi = np.dot(kalman_gain, np.dot(cov_terms, kalman_gain.T))
        P_c = estimate.cov_prior-np.dot(kalman_gain, np.dot(estimate.S, kalman_gain.T))
        estimate.cov_posterior = betas[-1]*estimate.cov_prior+(1-betas[-1])*P_c+soi


class Manager(object):
    def __init__(self, tracking_method, logger=DefaultLogger()):
        self.tracking_method = tracking_method
        self.logger = logger
        self.est_posterior = np.empty((4, 0))
        self.measurements_used = np.empty((2, 0))
        self.track_file = dict()
        self.active_tracks = set()

    def step(self, measurements):
        # Step active tracks
        timestamp = measurements[0].timestamp
        latest_estimates = [self.track_file[idx][-1] for idx in self.active_tracks]
        estimates, unused_measurements = self.tracking_method.step(latest_estimates, measurements, timestamp)
        self.update_track_file(estimates)
        self.latest_est = estimates[0]
        self.est_posterior = np.append(self.est_posterior, [[self.latest_est.est_posterior[0]], [self.latest_est.est_posterior[1]], [self.latest_est.est_posterior[2]], [self.latest_est.est_posterior[3]]], axis=1)
        if any(self.latest_est.measurements):
            self.measurements_used = np.append(self.measurements_used, np.array([measurement.value for measurement in self.latest_est.measurements]).T, axis=1)

    def update_track_file(self, estimates):
        [self.track_file[est.track_index].append(est) for est in estimates]

    def add_new_tracks(self, new_estimates):
        # Assumes that new_estimates = [[est_11, est_12, ...], [est_21, est_22, ...], ...]
        # which means that the initiation method outputs a historic estimate list
        for estimates in new_estimates:
            t_idx = estimates[0].track_index
            self.track_file[t_idx] = estimates
            self.active_tracks.add(t_idx)

    def ret_posterior(self):
        return self.est_posterior

    def ret_measurements(self):
        return self.measurements_used

