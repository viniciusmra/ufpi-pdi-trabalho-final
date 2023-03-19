from skimage.morphology import skeletonize, skeletonize_3d
import cv2
import numpy as np

from PIL import Image
import numpy
import cv2

from collections import deque

#Binariza a imagem usando o método de Otsu
def binarize(original_image):
    image = original_image.copy()
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    
    return binary_image


#Retorna todos os vizinhos com a mesma cor de um pixel
def getNeighbors(img, pixel):
    x, y = pixel
    level = img[x][y]
    neighbors = []

    queue = deque([(x, y)])
    visited = numpy.zeros_like(img, dtype=bool)
    visited[x][y] = True
    
    while len(queue) > 0:
        current_pixel = queue.popleft()
        qx, qy = current_pixel

        if(checkPixel(img, (qx - 1, qy - 1), level) and not visited[qx - 1][qy - 1]):
            queue.append((qx - 1, qy - 1))
            visited[qx - 1][qy - 1] = True
        
        if(checkPixel(img, (qx, qy - 1), level) and not visited[qx][qy - 1]):
            queue.append((qx, qy - 1))
            visited[qx][qy - 1] = True
        
        if(checkPixel(img, (qx + 1, qy - 1), level) and not visited[qx+1][qy-1]):
            queue.append((qx + 1, qy - 1))
            visited[qx+1][qy-1] = True
            
        if(checkPixel(img, (qx - 1, qy), level) and not visited[qx - 1][qy]):
            queue.append((qx - 1, qy))
            visited[qx - 1][qy] = True

        if(checkPixel(img, (qx + 1, qy), level) and not visited[qx + 1][qy]):
            queue.append((qx + 1, qy))
            visited[qx + 1][qy] = True

        if(checkPixel(img, (qx - 1, qy + 1), level) and not visited[qx - 1][qy + 1]):
            queue.append((qx - 1, qy + 1))
            visited[qx - 1][qy + 1] = True

        if(checkPixel(img, (qx, qy + 1), level) and not visited[qx][qy + 1]):
            queue.append((qx, qy + 1))
            visited[qx][qy + 1] = True
        
        if(checkPixel(img, (qx + 1, qy + 1), level) and not visited[qx + 1][qy + 1]):
            queue.append((qx + 1, qy + 1))
            visited[qx + 1][qy + 1] = True
        
        neighbors.append(current_pixel)
        
        img[qx][qy] = 200

    return neighbors

        
# checa se um determinado pixel de uma imagem tem valor "level"
def checkPixel(img, pixel, level):
    x, y = pixel
    max_x = len(img)
    max_y = len(img[0])

    if(x < 0 or y < 0 or x >= max_x or y >= max_y): #verifica se aquele pixel existe na imagem
        return False
    
    if(img[x][y] == level):
        return True
    else:
        return False

def clearBorder(original_img, color):
    img = original_img.copy()
    groups = []
    for x in range(0, len(img)):
        if(img[x][0] == 0):
            groups.append(getNeighbors(img, (x, 0)))

        if(img[x][len(img[0])-1] == 0):
            groups.append(getNeighbors(img, (x, len(img[0])-1)))
    
    for y in range(0, len(img[0])):
        if(img[0][y] == 0):
            groups.append(getNeighbors(img, (0, y)))

        if(img[len(img)-1][y] == 0):
            groups.append(getNeighbors(img, (len(img)-1, y)))
            
    for group in groups:
        for pixel in group:
            x, y = pixel
            img[x][y] = color

    return img

def colorizeGroup(original_image, group, grayscale):
    image = original_image.copy()

    for pixel in group:
            x, y = pixel
            image[x][y] = grayscale

    return image

def getGroups(origial_img):
    image = origial_img.copy()

    #new_image = img.copy()
    groups = []
    
    for x in range(0, len(image)):
        for y in range(0, len(image[0])):
            if(image[x][y] == 0):
                groups.append(getNeighbors(image, (x, y)))
                image = colorizeGroup(image, groups[-1], 1)

    return groups


def floodFill(original_img):
    gray = original_img.copy()

    # Criando uma máscara com o mesmo tamanho da imagem
    mask = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2), np.uint8)

    # Definindo o ponto inicial para a inundação (aqui estamos usando o ponto (0, 0))
    start_point = (0, 0)

    # Definindo a tolerância da inundação
    # Aqui estamos usando uma tolerância baixa de 10, mas você pode ajustar esse valor conforme necessário
    tolerance = 10

    # Chamando a função cv2.floodFill()
    cv2.floodFill(gray, mask, start_point, 0, tolerance)

    return original_img.copy() - gray

def getBounds(groups):
    groupBounds = []
    for group in groups:
        minY = group[0][1] 
        maxY = group[0][1] 
        minX = group[0][0]
        maxX = group[0][0] 
        for pixel in group:
            x, y = pixel
            if(x < minX):
                minX = x
            if(x > maxX):
                maxX = x
            if(y < minY):
                minY = y
            if(y > maxY):
                maxY = y
        groupBounds.append((minX, minY, maxX, maxY))

    return groupBounds

def createSubImage(group, groupSize):

    print(groupSize)

    width = groupSize[2] - groupSize[0]
    height = groupSize[3] - groupSize[1]

    minX = groupSize[0]
    minY = groupSize[1]

    image = np.ones((height, width), np.uint8) * 255
    print(len(image))
    print(len(image[0]))
    for pixel in group:
        x, y = pixel
        #print("(" + str(x-minX) + ", " + str(y - minY) + ")")
        image[x - minX][y - minY] = 0
	
    return image
