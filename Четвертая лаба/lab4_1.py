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


def treugolnik(x0, y0, z0, i0, x1, y1, z1, i1, x2, y2, z2, i2, Z_buf, img_mat):
    xmin = max(0, min(x0, x1, x2))
    ymin = max(0, min(y0, y1, y2))
    xmax = min(cols-1, max(x0, x1, x2))
    ymax = min(rows-1, max(y0, y1, y2))


    for x in range(int(xmin), int(xmax) + 1):
        for y in range(int(ymin), int(ymax) + 1):
            lambdas = baricentric(x, y, x0, y0, x1, y1, x2, y2)
            lambda0, lambda1, lambda2 = lambdas
            
            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                newZ = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                
                if newZ < Z_buf[x][y]:
                    intensity = lambda0 * i0 + lambda1 * i1 + lambda2 * i2
                    
                    color_value = max(0, min(255, int(255 * intensity)))
                    color = (color_value, color_value, color_value)
                    
                    img_mat[y, x] = color
                    Z_buf[x][y] = newZ

def vertex_normals(vertices, polygons):
    vertex_normals = [np.zeros(3) for _ in range(len(vertices))]
    vertex_weights = [0.0 for _ in range(len(vertices))]
    
    for poly in polygons:
        v0, v1, v2 = vertices[poly[0]-1], vertices[poly[1]-1], vertices[poly[2]-1]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        area = np.linalg.norm(normal)
        
        if area > 1e-10:
            normal = normal / area
            
            # нормаль
            for vertex_idx in poly:
                idx = vertex_idx - 1
                vertex_normals[idx] += normal * area
                vertex_weights[idx] += area
    
    for i in range(len(vertices)):
        if vertex_weights[i] > 1e-10:
            vertex_normals[i] /= vertex_weights[i]
            length = np.linalg.norm(vertex_normals[i])
            if length > 1e-10:
                vertex_normals[i] /= length
    
    return vertex_normals


polygons = []
spl = []
vertices = []
size_rabbit = 4000.0
size_x = 2000
size_y = 2000

alfa = np.radians(10)
betta = np.radians(45) 
gamma = np.radians(10)

R1 = np.array([[1, 0, 0], [0, np.cos(alfa), np.sin(alfa)], [0, -np.sin(alfa), np.cos(alfa)]])
R2 = np.array([[np.cos(betta),0, np.sin(betta)], [0, 1, 0], [-np.sin(betta), 0, np.cos(betta)]])
R3 = np.array([[np.cos(gamma), np.sin(gamma), 0], [-np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
R = np.matmul(np.matmul(R1, R2), R3)

for line in file:
    if line.startswith('v '):
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

print("Получил вершины")

vertex_normals = vertex_normals(vertices, polygons)

print("Normals")

light_direction = np.array([0, 0, -1])
light_direction = light_direction / np.linalg.norm(light_direction)

vertex_intensities = []
for i, normal in enumerate(vertex_normals):
    intensity = max(0, np.dot(normal, light_direction))
    vertex_intensities.append(intensity)

print("Vertex")

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

print("Project")

for i, poly in enumerate(polygons):
    if i % 1000 == 0:
        print(f"Отрисовано {i}/{len(polygons)} полигонов")
    
    v0_idx, v1_idx, v2_idx = poly[0]-1, poly[1]-1, poly[2]-1
    
    u0, v0, z0 = projected_vertices[v0_idx]
    u1, v1, z1 = projected_vertices[v1_idx]
    u2, v2, z2 = projected_vertices[v2_idx]
    
    i0 = vertex_intensities[v0_idx]
    i1 = vertex_intensities[v1_idx]
    i2 = vertex_intensities[v2_idx]
    
    treugolnik(u0, v0, z0, i0, u1, v1, z1, i1, u2, v2, z2, i2, Z_buf, img_mat)


img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img.png')
