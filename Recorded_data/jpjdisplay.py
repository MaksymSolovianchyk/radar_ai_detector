from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os
filename="recordingsv2/1m_jpg/waterfall_00.jpg"
img=Image.open(filename)
data=np.array(img)
plt.figure(figsize=(10,10))
plt.imshow(data,cmap="jet")
plt.colorbar(label="Intensity")
plt.title(os.path.basename(filename))
plt.axis("off")
plt.show()