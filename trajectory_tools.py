import numpy as np


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def dist(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)


class TrajectoryChange(object):
    def randomize_direction(self, state, varTheta=0.1):
        rho, theta = cart2pol(state[1], state[3])
        theta = theta + np.random.normal(scale=varTheta)
        x1, y1 = pol2cart(rho, theta)
        return np.array([[state[0]], [x1], [state[2]], [y1]])
