import math
import time
import numpy as np
import torch
from d2l import torch as d2l

n = 10000
a=torch.ones(n).reshape(100,100)
b=torch.ones(n).reshape(100,100)

class Timer:
    # 记录多次运行时间
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times)/len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()

c = torch.zeros(n).reshape(100,100)
timer = Timer()
for i in range(0,100):
    for j in range(0,100):
        c[i][j] = a[i][j] + b[i][j]

f'{timer.stop():.5f} sec'