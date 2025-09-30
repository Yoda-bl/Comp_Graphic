import numpy as np
from PIL import Image
from math import cos, sin, sqrt

img_mat = np.zeros((200, 200, 3), dtype = np.uint8) #uint8 - беззнаковая целое 8-бит

##def draw_line(img_mat, x0, y0, x1, y1):
##    step = 1.0 / 100
##    for t in np.arange(0, 1, step):
##        x = round((1.0 - t)*x0 + t*x1)
##        y = round((1.0 - t)*y0 + t*y1)
##        img_mat[y, x] = 255 

"""
def draw_line(img_mat, x0, y0, x1, y1):
    count = sqrt((x0-x1)**2 + (y0-y1)**2)
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t)*x0 + t*x1)
        y = round((1.0 - t)*y0 + t*y1)
        img_mat[y, x] = 255"""

"""
def draw_line(img_mat, x0, y0, x1, y1):
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t)*y0 + t*y1)
        img_mat[y, x] = 255 """

"""
def draw_line(img_mat, x0, y0, x1, y1):
    if (x0 > x1):
        x0, x1= x1, x0
        y0, y1 = y1, y0
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t)*y0 + t*y1)
        img_mat[y, x] = 255 
"""

"""def draw_line(img_mat, x0, y0, x1, y1):
    xChange = False
    if (abs(x0 - x1) < abs (y0 - y1)):
        x0, y0= y0, x0
        x1, y1= y1, x1
        xChange = True
    if (x0 > x1):
        x0, x1= x1, x0
        y0, y1 = y1, y0
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t)*y0 + t*y1)
        if (xChange):
            img_mat[x, y] = 255 
        else:
            img_mat[y, x] = 255 
"""

"""
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
    dy = abs(y1 - y0)/(x1 - x0)
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
        if (derror >0.5):
            derror -= 1.0
            y += y_update
"""

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

for k in range(13):
    x0, y0 = 100, 100
    x1 = int(100 + cos(2*3.14/13*k)*95)
    y1 = int(100 + sin(2*3.14/13*k)*95)
    draw_line(img_mat, x0, y0, x1, y1)

img = Image.fromarray(img_mat, mode = 'RGB')
img.save('img.png')
