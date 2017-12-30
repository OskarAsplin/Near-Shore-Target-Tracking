import numpy as np


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


class TrajectoryChange(object):
    def randomize_direction(self, state, varTheta=0.1):
        rho, theta = cart2pol(state[1], state[3])
        theta = theta + np.random.normal(scale=varTheta)
        x1, y1 = pol2cart(rho, theta)
        return np.array([[state[0]], [x1], [state[2]], [y1]])
