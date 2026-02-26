import os
import numpy as np
from matplotlib import cm
from PIL import Image
input_folder="recordingsv2/5m"
output_folder="recordingsv2/5m_jpg"
target_size=(224,224)
os.makedirs(output_folder,exist_ok=True)
for filename in sorted(os.listdir(input_folder)):
    if not filename.endswith(".npy"):
        continue
    in_path=os.path.join(input_folder,filename)
    data=np.load(in_path)
    mn,mx=float(data.min()),float(data.max())
    if mx>mn:
        x=(data-mn)/(mx-mn)
    else:
        x=np.zeros_like(data,dtype=np.float32)
    rgb=(cm.jet(x)[...,:3]*255).astype(np.uint8)
    img=Image.fromarray(rgb,mode="RGB")
    img=img.resize(target_size,resample=Image.BILINEAR)
    out_name=filename.replace(".npy",".jpg")
    out_path=os.path.join(output_folder,out_name)
    img.save(out_path,quality=95)
print("Done ✅")