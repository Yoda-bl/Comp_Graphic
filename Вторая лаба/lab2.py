import string
import random
import numpy as np
from PIL import Image, ImageOps
file = open("model_1.obj")

def baricentric(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2)*(y1 - y2) - (x1 - x2)*(y - y2)) / ((x0 - x2)*(y1 - y2) - (x1 - x2)*(y0 - y2))
    lambda1 = ((x0 - x2)*(y - y2) - (x - x2)*(y0 - y2)) / ((x0 - x2)*(y1 - y2) - (x1 - x2)*(y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return [lambda0, lambda1, lambda2]


def treugolnik(x0, y0, z0, x1, y1, z1, x2, y2, z2, Z_buf):
    xmin = min(x0, x1, x2)
    ymin = min(y0, y1, y2)
    zmin = min(z0, z1, z2)
    xmax = max(x0, x1, x2)
    ymax = max(y0, y1, y2)
    zmax = max(z0, z1, z2)
    if (xmin < 0): xmin = 0
    if (ymin < 0): ymin = 0
    if (zmin < 0): zmin = 0

    n = np.cross(np.array([x1 - x2, y1 - y2, z1 - z2]), np.array([x1 - x0, y1 - y0, z1 - z0]))
    cosA = np.dot(n, [0, 0, 1])/np.linalg.norm(n)
    if cosA >= 0: return

    color = (0, 0, -255*cosA)
    for x in range (int(xmin), int(xmax) + 1):
        for y in range(int(ymin), int(ymax) + 1):
            a = baricentric(x, y, x0, y0, x1, y1, x2, y2)
            if (a[0] >= 0 and a[1] >= 0 and a[2] >= 0):
                newZ = a[0]*z0 + a[1]*z1 +a[2]*z2
                if newZ < Z_buf[x][y]:
                    img_mat[y, x] = color
                    Z_buf[x][y] = newZ

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
rows, cols = 2000, 2000
Z_buf = [[np.inf for _ in range(cols)] for _ in range(rows)]

for i in range(len(polygons)):
    x0 = vertices[polygons[i][0] - 1][0] * 10000 + 750
    y0 = vertices[polygons[i][0] - 1][1] * 10000 + 750
    z0 = vertices[polygons[i][0] - 1][2] * 10000 + 750
    x1 = vertices[polygons[i][1] - 1][0] * 10000 + 750
    y1 = vertices[polygons[i][1] - 1][1] * 10000 + 750
    z1 = vertices[polygons[i][1] - 1][2] * 10000 + 750
    x2 = vertices[polygons[i][2] - 1][0] * 10000 + 750
    y2 = vertices[polygons[i][2] - 1][1] * 10000 + 750
    z2 = vertices[polygons[i][2] - 1][2] * 10000 + 750
    treugolnik(x0, y0, z0, x1, y1, z1, x2, y2, z2, Z_buf)

img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img.png')
