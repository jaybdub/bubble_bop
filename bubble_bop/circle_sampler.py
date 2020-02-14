import numpy as np


class CircleSampler(object):
    
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        
    def sample(self):
        """Samples uniformily from circle"""
        t = 2.0 * np.pi * np.random.random()
        u = np.random.random() + np.random.random()
        r = 2 - u if u > 1 else u
        x, y = [r*np.cos(t), r*np.sin(t)]
        x, y = self.radius * x, self.radius * y
        x, y = self.center[0] + x, self.center[1] + y
        return (x, y)
    
    def __repr__(self):
        return self.__dict__.__repr__()