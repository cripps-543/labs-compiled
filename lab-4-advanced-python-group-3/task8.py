import numpy as np
import matplotlib.pyplot as plt

sin = np.load("task7_sin.npy")
cos = np.load("task7_cos.npy")

plt.plot(sin, label="Sin")
plt.plot(cos, label="Cos")

plt.legend()

plt.show()
