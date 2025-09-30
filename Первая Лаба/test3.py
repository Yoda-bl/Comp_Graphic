import numpy as np
from PIL import Image, ImageOps
file = open("model_1.obj")
vertices = []
spl = []
for line in file:
    if line.startswith('v '):  # вершины
        spl = list(map(float, line.split()[1:4]))
        vertices.append(spl)

img_mat = np.zeros((1000, 1000, 3), dtype = np.uint8) #uint8 - беззнаковая целое 8-бит

for i in range(len(vertices)):
    img_mat[350 + round(4000 * vertices[i][1]), 500 + round(4000 * vertices[i][0])] = 155


img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img.png')