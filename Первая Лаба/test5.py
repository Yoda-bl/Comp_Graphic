import string
import numpy as np
from PIL import Image, ImageOps
file = open("model_1.obj")

def draw_line(img_mat, x0, y0, x1, y1):
    xChange = False
    if (abs(x0 - x1) < abs (y0 - y1)):
        x0, y0= y0, x0
        x1, y1= y1, x1
        xChange = True
    if (x0 > x1):
        x0, x1= x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2 * abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t)*y0 + t*y1)
        if (xChange):
            img_mat[x, y] = 255 
        else:
            img_mat[y, x] = 255 
        derror += dy
        if (derror > (x1- x0)):
            derror -= 2 * (x1- x0)
            y += y_update

polygons = []
spl = []
vertices = []




for line in file:
    if line.startswith('v '):  # вершины
        spl = list(map(float, line.split()[1:4]))
        vertices.append(spl)

file.close()
file = open("model_1.obj")

for line in file:
    if line.startswith('f '):  # вершины
        spl = [int(x.split('/')[0]) for x in line.split()[1:4]]

        polygons.append(spl)

img_mat = np.zeros((2000, 2000, 3), dtype = np.uint8) #uint8 - беззнаковая целое 8-бит

for i in range(len(polygons)):
    x0 = int(vertices[polygons[i][0] - 1][0] * 10000 + 1050)
    y0 = int(vertices[polygons[i][0] - 1][1] * 10000 + 350)
    x1 = int(vertices[polygons[i][1] - 1][0] * 10000 + 1050)
    y1 = int(vertices[polygons[i][1] - 1][1] * 10000 + 350)
    x2 = int(vertices[polygons[i][2] - 1][0] * 10000 + 1050)
    y2 = int(vertices[polygons[i][2] - 1][1] * 10000 + 350)
    draw_line(img_mat, x0, y0, x1, y1)
    draw_line(img_mat, x1, y1, x2, y2)
    draw_line(img_mat, x0, y0, x2, y2)



img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img.png')