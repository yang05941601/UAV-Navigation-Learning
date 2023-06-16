import numpy as np


class User(object):
    def __init__(self, x, y,z):
        self.x = x
        self.y = y
        self.z = z


class UAV(object):
    def __init__(self, x, y, h):
        self.x = x
        self.y = y
        self.h = h

    def move_inside_test(self, phi, dist_max):
        phi = np.pi * phi
        dist = dist_max
        self.x = self.x + dist * np.cos(phi)
        self.y = self.y + dist * np.sin(phi)


