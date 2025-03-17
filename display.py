import os
import numpy as np
from PIL import Image

# 이미지 열기
img = Image.open('mask_gray.png')

label_image_array = np.array(img)

print(label_image_array.shape)