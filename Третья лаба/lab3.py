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
    xmin = max(0, min(x0, x1, x2))
    ymin = max(0, min(y0, y1, y2))
    xmax = min(cols-1, max(x0, x1, x2))
    ymax = min(rows-1, max(y0, y1, y2))


    n = np.cross(np.array([x1 - x2, y1 - y2, z1 - z2]), np.array([x1 - x0, y1 - y0, z1 - z0]))
    cosA = np.dot(n, [0, 0, 1])/np.linalg.norm(n)
    if cosA >= 0: return

    color = (-255*cosA, 0, -255*cosA)
    for x in range (int(xmin), int(xmax) + 1):
        for y in range(int(ymin), int(ymax) + 1):
            a = baricentric(x, y, x0, y0, x1, y1, x2, y2)
            if (a[0] >= 0 and a[1] >= 0 and a[2] >= 0):
                newZ = a[0]*z0 + a[1]*z1 + a[2]*z2
                if newZ < Z_buf[x][y]:
                    img_mat[y, x] = color
                    Z_buf[x][y] = newZ

polygons = []
spl = []
vertices = []
size_rabbit = 3000.0
size_x = 2000
size_y = 2000

# Преобразование в радианы
alfa = np.radians(0)
betta = np.radians(0) 
gamma = np.radians(0)

R1 = np.array([[1, 0, 0], [0, np.cos(alfa), np.sin(alfa)], [0, -np.sin(alfa), np.cos(alfa)]])
R2 = np.array([[np.cos(betta),0, np.sin(betta)], [0, 1, 0], [-np.sin(betta), 0, np.cos(betta)]])
R3 = np.array([[np.cos(gamma), np.sin(gamma), 0], [-np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
R = np.matmul(np.matmul(R1, R2), R3)

#t = np.array([size_x/2, size_y/3, 500])

for line in file:
    if line.startswith('v '):  # вершины
        spl = list(map(float, line.split()[1:4]))
        spl = np.array(spl) * size_rabbit
        spl = R @ spl
        spl[2] += 500
        vertices.append(spl)

file.close()
file = open("model_1.obj")

for line in file:
    if line.startswith('f '):  # вершины
        spl = [int(x.split('/')[0]) for x in line.split()[1:4]]

        polygons.append(spl)
file.close()

img_mat = np.zeros((size_x, size_y, 3), dtype = np.uint8) #uint8 - беззнаковая целое 8-бит
rows, cols = size_x, size_y
Z_buf = np.full((size_x, size_y), np.inf)

aX = 800   # масштаб по X
aY = 800   # масштаб по Y
u0 = size_x / 2  # центр изображения по X
v0 = size_y / 2  # центр изображения по Y

projected_vertices = []
for v in vertices:
    X, Y, Z = v
    if Z <= 0:
        Z = 1e-5
    u = aX * X / Z + u0
    v_proj = aY * Y / Z + v0
    projected_vertices.append([u, v_proj, Z])

for i in range(len(polygons)):
    Z0 = vertices[polygons[i][0] - 1][2]
    Z1 = vertices[polygons[i][1] - 1][2]
    Z2 = vertices[polygons[i][2] - 1][2]

    u_a, v_a, _ = projected_vertices[polygons[i][0] - 1]
    u_b, v_b, _ = projected_vertices[polygons[i][1] - 1]
    u_c, v_c, _ = projected_vertices[polygons[i][2] - 1]

    treugolnik(u_a, v_a, Z0, u_b, v_b, Z1, u_c, v_c, Z2, Z_buf)

img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img.png')
