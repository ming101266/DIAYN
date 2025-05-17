import numpy as np

class Point2DEnv:
    def __init__(self, max_step=0.1, bounds=1.0):
        self.state = None
        self.max_step = max_step
        self.bounds = bounds

    def reset(self):
        self.state = np.zeros(2)
        return self.state.copy()

    def step(self, action):
        action = np.clip(action, -self.max_step, self.max_step)
        self.state += action
        self.state = np.clip(self.state, -self.bounds, self.bounds)
        return self.state.copy()