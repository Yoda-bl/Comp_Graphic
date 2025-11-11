import string
import random
import numpy as np
from PIL import Image, ImageOps

def baricentric(x, y, x0, y0, x1, y1, x2, y2):
    denominator = ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    if abs(denominator) < 1e-10:
        return [0, 0, 0]
    
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / denominator
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / denominator
    lambda2 = 1.0 - lambda0 - lambda1
    return [lambda0, lambda1, lambda2]

def treugolnik(x0, y0, z0, u0_tex, v0_tex, i0,
               x1, y1, z1, u1_tex, v1_tex, i1,
               x2, y2, z2, u2_tex, v2_tex, i2,
               Z_buf, img_mat, texture_img):
    xmin = max(0, min(x0, x1, x2))
    ymin = max(0, min(y0, y1, y2))
    xmax = min(cols-1, max(x0, x1, x2))
    ymax = min(rows-1, max(y0, y1, y2))

    tex_width, tex_height = texture_img.size

    for x in range(int(xmin), int(xmax) + 1):
        for y in range(int(ymin), int(ymax) + 1):
            lambdas = baricentric(x, y, x0, y0, x1, y1, x2, y2)
            lambda0, lambda1, lambda2 = lambdas
            
            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                newZ = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                
                if newZ < Z_buf[x][y]:
                    u_tex = lambda0 * u0_tex + lambda1 * u1_tex + lambda2 * u2_tex
                    v_tex = lambda0 * v0_tex + lambda1 * v1_tex + lambda2 * v2_tex
                    
                    tex_x = int(u_tex * (tex_width - 1))
                    tex_y = int((1 - v_tex) * (tex_height - 1))
                    
                    tex_x = max(0, min(tex_width - 1, tex_x))
                    tex_y = max(0, min(tex_height - 1, tex_y))
                    
                    texture_color = texture_img.getpixel((tex_x, tex_y))
                    
                    intensity = lambda0 * i0 + lambda1 * i1 + lambda2 * i2
                    if intensity < 0: intensity = 0
                    
                    r = int(texture_color[0] * intensity)
                    g = int(texture_color[1] * intensity)
                    b = int(texture_color[2] * intensity)
                    
                    color = [r, g, b]
                    
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

file = open("model_1.obj")

polygons = []
texture_coords = []
polygon_textures = []
spl = []
vertices = []
size_rabbit = 2000.0
size_x = 2000
size_y = 2000

texture_img = Image.open("bunny-atlas.jpg") 

alfa = np.radians(10)
betta = np.radians(182) 
gamma = np.radians(0)

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
    elif line.startswith('vt '): 
        tex_coords = list(map(float, line.split()[1:3]))
        texture_coords.append(tex_coords)

file.close()
file = open("model_1.obj")

for line in file:
    if line.startswith('f '): 
        face_data = line.split()[1:4]
        vertex_indices = []
        texture_indices = []
        
        for face_part in face_data:
            parts = face_part.split('/')
            vertex_indices.append(int(parts[0]))
            if len(parts) > 1 and parts[1]:
                texture_indices.append(int(parts[1]))
        
        polygons.append(vertex_indices)
        polygon_textures.append(texture_indices)

file.close()

vertex_normals_list = vertex_normals(vertices, polygons)

light_direction = np.array([0, 0, -1])
light_direction = light_direction / np.linalg.norm(light_direction)

vertex_intensities = []
for i, normal in enumerate(vertex_normals_list):
    intensity = max(0, np.dot(normal, light_direction))
    vertex_intensities.append(intensity)

img_mat = np.zeros((size_x, size_y, 3), dtype=np.uint8)
rows, cols = size_x, size_y
Z_buf = np.full((size_x, size_y), np.inf)

aX = 2000   # масштаб по X
aY = 2000   # масштаб по Y
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

for i, poly in enumerate(polygons):    
    v0_idx, v1_idx, v2_idx = poly[0]-1, poly[1]-1, poly[2]-1
    tex0_idx, tex1_idx, tex2_idx = polygon_textures[i]
    
    u0_proj, v0_proj, z0 = projected_vertices[v0_idx]
    u1_proj, v1_proj, z1 = projected_vertices[v1_idx]
    u2_proj, v2_proj, z2 = projected_vertices[v2_idx]
    
    u0_tex, v0_tex = texture_coords[tex0_idx - 1] if tex0_idx > 0 else texture_coords[v0_idx]
    u1_tex, v1_tex = texture_coords[tex1_idx - 1] if tex1_idx > 0 else texture_coords[v1_idx]
    u2_tex, v2_tex = texture_coords[tex2_idx - 1] if tex2_idx > 0 else texture_coords[v2_idx]
    
    i0 = vertex_intensities[v0_idx]
    i1 = vertex_intensities[v1_idx]
    i2 = vertex_intensities[v2_idx]
    
    treugolnik(u0_proj, v0_proj, z0, u0_tex, v0_tex, i0,
                                     u1_proj, v1_proj, z1, u1_tex, v1_tex, i1,
                                     u2_proj, v2_proj, z2, u2_tex, v2_tex, i2,
                                     Z_buf, img_mat, texture_img)

img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img_2.png')