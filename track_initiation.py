import numpy as np
from scipy.linalg import block_diag
import tracking


class MOfNInitiation(object):
    def __init__(self, M_required, N_test, tracking_method, track_gate):
        self.M_required = M_required
        self.N_test = N_test
        self.tracking_method = tracking_method
        self.initiator_gate = track_gate
        self.initiators = []
        self.preliminary_tracks = dict()
        self.track_status = dict()
        self.current_index = 1

    def step(self, measurements, timestamp):
        preliminary_estimates = self.get_preliminary_estimates()
        estimates, unused_measurements = self.tracking_method.step(preliminary_estimates, measurements, timestamp)
        [self.preliminary_tracks[estimate.track_index].append(estimate) for estimate in estimates]
        confirmed_estimates = self.update_tracks(estimates)
        # Use the unused measurements to form new tracks
        measurement_used = [False for _ in unused_measurements]
        for measurement_idx, measurement in enumerate(unused_measurements):
            for initiator in self.initiators:
                if self.initiator_gate.gate_measurement(initiator, measurement):
                    measurement_used[measurement_idx] = True
                    new_estimates = self.form_track(initiator, measurement)
                    self.preliminary_tracks[self.current_index] = new_estimates
                    self.track_status[self.current_index] = {'M' : 0, 'N' : 0}
                    self.current_index += 1
        self.initiators = [z for z, used in zip(unused_measurements, measurement_used) if not used]
        return confirmed_estimates

    def update_tracks(self, estimates):
        confirmed_estimates = []
        for estimate in estimates:
            t_idx = estimate.track_index
            self.track_status[t_idx]['N'] += 1
            if len(estimate.measurements) > 0:
                self.track_status[t_idx]['M'] += 1
            M = self.track_status[t_idx]['M']
            N = self.track_status[t_idx]['N']
            if N <= self.N_test and M >= self.M_required:
                confirmed_estimates.append(self.preliminary_tracks[t_idx])
                del self.track_status[t_idx]
            elif N >= self.N_test and M < self.M_required:
                del self.track_status[t_idx]
                del self.preliminary_tracks[t_idx]
        return confirmed_estimates

    def get_preliminary_estimates(self):
        return [estimates[-1] for track_id, estimates in self.preliminary_tracks.items() if track_id in self.track_status.keys()]

    def form_track(self, initiator, measurement):
        H = initiator.measurement_mapping
        R1 = initiator.covariance
        R2 = measurement.covariance
        t1 = initiator.timestamp
        t2 = measurement.timestamp
        dt = t2-t1
        F, _ = self.tracking_method.target_model.get_model(dt)
        H_s = np.vstack((H, np.dot(H,F)))
        z_s = np.hstack((initiator.value, measurement.value))
        R_s = block_diag(R1, R2)
        S_s = np.dot(H_s.T, np.dot(np.linalg.inv(R_s), H_s))
        S_s_inv = np.linalg.inv(S_s)
        est_x1 = np.dot(np.dot(S_s_inv, np.dot(H_s.T, np.linalg.inv(R_s))), z_s)
        est_x2 = np.dot(F, est_x1)
        cov_x1 = S_s_inv
        cov_x2 = np.dot(F, np.dot(S_s_inv, F.T))
        est_1 = tracking.Estimate(t1, est_x1, cov_x1, is_posterior=True, track_index = self.current_index)
        est_2 = tracking.Estimate(t2, est_x2, cov_x2, is_posterior=True, track_index = self.current_index)
        est_1.store_measurement(initiator)
        est_2.store_measurement(measurement)
        return [est_1, est_2]
    
    def get_preliminary_tracks(self):
        return self.preliminary_tracks
