import numpy as np
import tracking


class Radar(object):
    def __init__(self, radar_range, clutter_density, detection_probability=1, measurement_covariance=0):
        self.radar_range = radar_range
        self.area = self.calculate_area(radar_range)
        self.clutter_density = clutter_density
        self.detection_probability = detection_probability
        self.measurement_covariance = measurement_covariance
        self.clutter_map = None

    def generate_clutter_measurements(self, timestamp):
        if self.clutter_map == None:
            measurements = []
            if self.clutter_density > 0:
                number_of_measurements = np.random.poisson(self.area*self.clutter_density)
                # Rejection sampling
                while len(measurements) < number_of_measurements:
                    measurement = np.random.uniform(-self.radar_range, self.radar_range, 2)
                    if self.inside_range(measurement):
                        measurements.append(tracking.Measurement(measurement, timestamp, self.measurement_covariance))
        else:
            measurements_loc = self.clutter_map.generate_clutter()
            measurements = [tracking.Measurement(measurement, timestamp, self.measurement_covariance) for measurement in measurements_loc]
        return measurements

    def generate_target_measurements(self, true_positions, timestamp): # Timestamp should probably be incorporated in the position
        measurements = []
        for target in true_positions:
            if np.any(self.measurement_covariance > 0):
                noise = np.random.multivariate_normal(np.zeros(2), self.measurement_covariance)
            else:
                noise = np.zeros(2)
            measurement = target + noise
            is_detected = np.random.uniform() < self.detection_probability
            if is_detected and self.inside_range(measurement):
                measurements.append(tracking.Measurement(measurement, timestamp, self.measurement_covariance))
        return measurements

    def generate_measurements(self, true_positions, timestamp):
        measurements_targets = self.generate_target_measurements(true_positions, timestamp)
        measurements_clutter = self.generate_clutter_measurements(timestamp)
        return measurements_targets+measurements_clutter

    def add_clutter_map(self, clutter_map):
        self.clutter_map = clutter_map


class SquareRadar(Radar):
    def inside_range(self, measurement):
        return np.linalg.norm(measurement, np.inf) < self.radar_range

    def calculate_area(self, radar_range):
        return np.pi*radar_range**2


class CircularRadar(Radar):
    def inside_range(self, measurement):
        return np.linalg.norm(measurement, 2) < self.radar_range

    def calculate_area(self, radar_range):
        return 4*radar_range**2
