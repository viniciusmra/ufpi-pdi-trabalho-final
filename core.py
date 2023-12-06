import math
from skimage.morphology import skeletonize
from scipy.spatial import distance
import cv2
import numpy as np

import numpy
import cv2

from collections import deque

#Binariza a imagem usando o método de Otsu
def binarize(rawImage):
    img = rawImage.copy()
    _, binary_image = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    
    return binary_image


# Retorna uma lista com todos os pixels conectados com vizinhaça 8 com mesma intesidade de um pixel inicial
def getNeighbors(img, pixel):
    x, y = pixel
    level = img[y][x]
    neighbors = []

    queue = deque([(x, y)])
    visited = numpy.zeros_like(img, dtype=bool)
    visited[y][x] = True
    
    while len(queue) > 0:
        currentPixel = queue.popleft()
        qx, qy = currentPixel

        if(checkPixel(img, (qx - 1, qy - 1), level) and not visited[qy - 1][qx - 1]):
            queue.append((qx - 1, qy - 1))
            visited[qy - 1][qx - 1] = True
        
        if(checkPixel(img, (qx, qy - 1), level) and not visited[qy - 1][qx]):
            queue.append((qx, qy - 1))
            visited[qy - 1][qx] = True
        
        if(checkPixel(img, (qx + 1, qy - 1), level) and not visited[qy - 1][qx + 1]):
            queue.append((qx + 1, qy - 1))
            visited[qy - 1][qx + 1] = True
            
        if(checkPixel(img, (qx - 1, qy), level) and not visited[qy][qx - 1]):
            queue.append((qx - 1, qy))
            visited[qy][qx - 1] = True

        if(checkPixel(img, (qx + 1, qy), level) and not visited[qy][qx + 1]):
            queue.append((qx + 1, qy))
            visited[qy][qx + 1] = True

        if(checkPixel(img, (qx - 1, qy + 1), level) and not visited[qy + 1][qx - 1]):
            queue.append((qx - 1, qy + 1))
            visited[qy + 1][qx - 1] = True

        if(checkPixel(img, (qx, qy + 1), level) and not visited[qy + 1][qx]):
            queue.append((qx, qy + 1))
            visited[qy + 1][qx] = True
        
        if(checkPixel(img, (qx + 1, qy + 1), level) and not visited[qy + 1][qx + 1]):
            queue.append((qx + 1, qy + 1))
            visited[qy + 1][qx + 1] = True
        
        neighbors.append(currentPixel)

    return neighbors

        
# Checa se o pixel existe (está dentro dos limites de imagem) e se ele tem uma determinada intesidade "value" (retorna True ou False)
def checkPixel(img, pixel, value):
    x, y = pixel
    maxY = len(img)
    maxX = len(img[0])

    if(x < 0 or y < 0 or x >= maxX or y >= maxY): # verifica se o pixel está dentro dos limites da imagem
        return False
    
    if(img[y][x] == value): # verifica se o pixel tem uma determinada intesidade
        return True
    else:
        return False

# Percorre as bordas da imagem e, caso ache um pixel preto, identifica aquele grupo de pixels e pinta com a cor desejada
# Retorna uma imagem sem os objetos da borda
def getBorderObjects(rawImage):
    img = rawImage.copy()
    borderObjects = []
    for y in range(0, len(img)):
        if(img[y][0] == 0):
            borderObjects.append(getNeighbors(img, (0, y)))
            img = colorizeObject(img, borderObjects[-1], 1)

        if(img[y][len(img[0]) - 1] == 0):
            borderObjects.append(getNeighbors(img, (len(img[0]) - 1, y)))
            img = colorizeObject(img, borderObjects[-1], 1)
    
    for x in range(0, len(img[0])):
        if(img[0][x] == 0):
            borderObjects.append(getNeighbors(img, (x, 0)))
            img = colorizeObject(img, borderObjects[-1], 1)

        if(img[len(img) - 1][x] == 0):
            borderObjects.append(getNeighbors(img, (x, len(img) - 1)))
            img = colorizeObject(img, borderObjects[-1], 1)

    return borderObjects

# Recebe um grupo de pixels e aplica uma cor a todos os pixel do grupo. Retorna a imagem modificada
def colorizeObject(rawImage, object, value):
    img = rawImage.copy()

    for pixel in object:
            x, y = pixel
            img[y][x] = value #ok

    return img

def getObjects(rawImage):
    img = rawImage.copy()

    objects = []
    
    for y in range(0, len(img)):
        for x in range(0, len(img[0])):
            if(img[y][x] == 0):
                objects.append(getNeighbors(img, (x, y)))
                img = colorizeObject(img, objects[-1], 1)

    return objects

# retorna uma nova imagem sem buracos internos aos objetos
def removeHoles(rawImage):
    img = rawImage.copy()

    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8) # Cria uma máscara com o mesmo tamanho da imagem

    start_point = (0, 0)    # Define o ponto de início, no caso, o ponto 0,0 da imagem
    tolerance = 10          # Define a tolerância da inundaçao

    cv2.floodFill(img, mask, start_point, 0, tolerance) # Chamando a função cv2.floodFill()

    return rawImage.copy() - img # Retorna a imagem original - a inundação

# Retorna os pontos limitantes de cada objeto (x e y máximos e mínimos)
def getBounds(objects):
    objectBounds = []
    for object in objects:
        minX = object[0][0]
        maxX = object[0][0] 
        minY = object[0][1] 
        maxY = object[0][1] 
       
        for pixel in object:
            x, y = pixel
            if(x < minX):
                minX = x
            if(x > maxX):
                maxX = x
            if(y < minY):
                minY = y
            if(y > maxY):
                maxY = y
        objectBounds.append((minX, minY, maxX, maxY))

    return objectBounds

# Retorna uma subimagem conteando apenas o objeto, com folga de um 1 pixel para cada lado
def createSubImage(object, objectBounds):

    width = objectBounds[2] - objectBounds[0] + 3
    height = objectBounds[3] - objectBounds[1] + 3

    minX = objectBounds[0]
    minY = objectBounds[1]

    image = np.ones((height, width), np.uint8) * 255

    for pixel in object:
        x, y = pixel
        image[y - minY + 1][x - minX + 1] = 0

    return image

# Retorna uma lista com as pontas os esqueletos dos objetos
def getTips(skeleton):
    window = np.ones((3, 3), dtype=np.uint8)
    tips = []

    for i in range(1, len(skeleton[0])):  # x
        for j in range(2, len(skeleton)): # y
            if(skeleton[j][i] == 0):
                window = skeleton[j-1:j+2, i-1:i+2]
                value = np.sum(np.logical_not(window))
                if(value == 2):
                    tips.append((i, j))
    return tips

# Retorna uma tupla com o ponto central do esqueleto do objeto
def getCenter(skeleton):
    window = np.ones((3, 3), dtype=np.uint8)
    center = (0,0)
    max = 0
    for i in range(1, len(skeleton[0])):  # x
        for j in range(2, len(skeleton)): # y
            if(skeleton[j][i] == 0):
                window = skeleton[j-1:j+2, i-1:i+2]
                value = np.sum(np.logical_not(window))
                if(value > max):
                    center = (i, j)
                    max = value
    return center

# Retorna a excentricidade do objeto com base na posição do centro e das pontas do esqueleto
def getEccentricity(img):
    tips = getTips(img)
    center = getCenter(img)

    tipsDistance = []
    
    for tip in tips:
        tipsDistance.append(distance.euclidean(center, tip))
    rMax = np.max(tipsDistance)
    rMin = np.min(tipsDistance)
    e = (rMax - rMin) / (rMax + rMin)
    return e
    

def checkObjects(objects, objectBounds):
    approvedGears = [False for i in range(len(objects))]
    reprovedGears = [False for i in range(len(objects))]
    undefinedGears = [False for i in range(len(objects))]
    gearsTeeths = [0 for i in range(len(objects))]

    for index, object in enumerate(objects):
        subimg = createSubImage(object, objectBounds[index])
        
        subimg = removeHoles(subimg) # fechamento
        skeleton = skeletonize(cv2.bitwise_not(subimg))
        skeleton = cv2.bitwise_not(skeleton.astype('uint8') * 255)
        
        e = getEccentricity(skeleton)
        if(e > 0.2):
            undefinedGears[index] = True
            
        else:
            approvedGears[index] = checkAngles(skeleton)
            reprovedGears[index] = not checkAngles(skeleton)
            gearsTeeths[index] = getNumberOfTeeths(skeleton)

    return approvedGears, reprovedGears, undefinedGears, gearsTeeths

def trimTips(rawSkeleton):
    skeleton = rawSkeleton.copy()
    center = getCenter(skeleton)
    r = center[0]
    for x in range(0, len(skeleton[0])):
            for y in range(0, len(skeleton)):
                if(skeleton[y][x] == 0):
                    d =  math.sqrt((x - center[0])**2 + (y - center[1])**2)
                    if(d > r/2):
                        skeleton[y][x] = 255

    return skeleton

# Retorna o número de dentes das engrenanges
def getNumberOfTeeths(skeleton):
    trimedSkeleton = trimTips(skeleton)
    tips = getTips(trimedSkeleton)
    return len(tips)

def checkAngles(skeleton):
    trimedSkeleton = trimTips(skeleton)
    tips = getTips(trimedSkeleton)
    center = getCenter(trimedSkeleton)
    expectedAngle = 360/len(tips)
    angles = []

    for tip in tips:
        h = (1,0)
        v = (tip[0] - center[0], tip[1] - center[1])

        # Calcula o coseno do ângulo entre a e b
        angle = np.degrees(np.arccos(np.dot(v, h) / (np.linalg.norm(v) * np.linalg.norm(h))))
        if(v[1] < 0):
            angle = 360 - angle
        angles.append(angle)

    angles.sort()
    
    for i in range(0, len(angles)-1):
        if(angles[i+1] - angles[i] > 1.2 * expectedAngle):
            return False
    if((360 + angles[0]) - angles[-1] > 1.2 * expectedAngle):
        return False
    return True

# exibe uma janela com a imagem
def show(img):
    cv2.imshow('Resultado', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
