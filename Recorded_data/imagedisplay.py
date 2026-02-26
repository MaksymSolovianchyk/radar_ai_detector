from matplotlib import pyplot as plt
import numpy as np
filename = "recordingsv2//1m//waterfall_00.npy"
data = np.load(filename)
plt.figure(figsize=(10,10))
plt.imshow(data, interpolation='nearest',aspect='auto')
plt.show()