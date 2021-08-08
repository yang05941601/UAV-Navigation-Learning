import numpy as np


class User(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class UAV(object):
    def __init__(self, x, y, h):
        self.x = x
        self.y = y
        self.h = h

    def move_inside_test(self, phi, dist_half,dist_max):
        phi = phi
        dist = dist_half + dist_max/2
        self.x = self.x + dist * np.cos(phi)
        self.y = self.y + dist * np.sin(phi)


