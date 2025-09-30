import numpy as np
from PIL import Image

img_mat = np.zeros((600,800, 3), dtype = np.uint8) #uint8 - беззнаковая целое 8-бит

for i in range(600):
    for j in range(800):
        img_mat[i,j] = (i+j)%256

img = Image.fromarray(img_mat, mode = 'RGB')
img.save('img.png')
