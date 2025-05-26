import random
import numpy as np
import time
import matplotlib.pyplot as plt

class performanceTracker:
    def __init__(self, function_names=None):
        if function_names is None:
            function_names = []
        self.function_names = function_names
        self.times = {name: 0.0 for name in function_names}
        self.call_counts = {name: 0 for name in function_names}
        self.history = {name: [] for name in function_names}
        self.idxs = []
    def start(self, function_name):
        if function_name not in self.function_names:
            raise ValueError(f"Function '{function_name}' not tracked.")
        self.start_time = time.perf_counter()
        self.current_function = function_name

    def end(self):
        if not hasattr(self, 'start_time'):
            raise RuntimeError("Timer was not started.")
        elapsed_time = time.perf_counter() - self.start_time
        self.times[self.current_function] += elapsed_time
        self.call_counts[self.current_function] += 1
        del self.start_time
        del self.current_function

    def get_average_time(self, function_name):
        if function_name not in self.function_names:
            raise ValueError(f"Function '{function_name}' not tracked.")
        if self.call_counts[function_name] == 0:
            return 0.0
        return self.times[function_name] / self.call_counts[function_name]
    
    def print_times(self):
        for name in self.function_names:
            avg_time = self.get_average_time(name)
            print(f"{name} averaged {avg_time:.6f} seconds/call (called {self.call_counts[name]} times)")
        return
    
    def reset(self):
        self.times = {name: 0.0 for name in self.function_names}
        self.call_counts = {name: 0 for name in self.function_names}
        self.history = {name: [] for name in self.function_names}
        return
    
    def historize(self, iteration_index):
        for name in self.function_names:
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(self.get_average_time(name))
            self.times[name] = 0.0
            self.call_counts[name]=0
        self.idxs.append(iteration_index)
        return
    
    def get_history(self, function_name):
        if function_name not in self.function_names:
            raise ValueError(f"Function '{function_name}' not tracked.")
        return self.history[function_name]
    
    def plot_history(self, function_name):
        import matplotlib.pyplot as plt
        if function_name not in self.function_names:
            raise ValueError(f"Function '{function_name}' not tracked.")
        plt.plot(self.idxs, self.history[function_name], label=function_name)
        plt.title(f"Performance History for {function_name}")
        plt.xlabel("Calls")
        plt.ylabel("Average Time (seconds)")
        plt.show()
        return
    
    def plot_all_histories(self):
        import matplotlib.pyplot as plt
        for name in self.function_names:
            plt.plot(self.idxs, self.history[name],  label=name)
        plt.title("Performance History for All Functions")
        plt.xlabel("iteration_index")
        plt.ylabel("Average Time (seconds)")
        plt.legend()
        plt.show()
        return
    