import math
from skimage.morphology import skeletonize, skeletonize_3d
from scipy.spatial import distance
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
    level = img[y][x]
    neighbors = []

    queue = deque([(x, y)])
    visited = numpy.zeros_like(img, dtype=bool)
    visited[y][x] = True
    
    while len(queue) > 0:
        current_pixel = queue.popleft()
        qx, qy = current_pixel

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
        
        neighbors.append(current_pixel)
        
        #img[qx][qy] = 200

    return neighbors

        
# checa se um determinado pixel de uma imagem tem valor "level"
def checkPixel(img, pixel, level):
    x, y = pixel
    max_y = len(img)
    max_x = len(img[0])

    if(x < 0 or y < 0 or x >= max_x or y >= max_y): #verifica se aquele pixel existe na imagem
        return False
    
    if(img[y][x] == level):
        return True
    else:
        return False

def clearBorder(original_img, color):
    img = original_img.copy()
    groups = []
    for y in range(0, len(img)):
        if(img[y][0] == 0):
            groups.append(getNeighbors(img, (0, y)))
            img = colorizeGroup(img, groups[-1], color)

        if(img[y][len(img[0]) - 1] == 0):
            groups.append(getNeighbors(img, (len(img[0]) - 1, y)))
            img = colorizeGroup(img, groups[-1], color)
    
    for x in range(0, len(img[0])):
        if(img[0][x] == 0):
            groups.append(getNeighbors(img, (x, 0)))
            img = colorizeGroup(img, groups[-1], color)

        if(img[len(img) - 1][x] == 0):
            groups.append(getNeighbors(img, (x, len(img) - 1)))
            img = colorizeGroup(img, groups[-1], color)
            
    for group in groups:
        for pixel in group:
            x, y = pixel
            img[y][x] = color

    return img

def colorizeGroup(original_image, group, grayscale):
    image = original_image.copy()

    for pixel in group:
            x, y = pixel
            image[y][x] = grayscale #ok

    return image

def getGroups(origial_img):
    image = origial_img.copy()

    #new_image = img.copy()
    groups = []
    
    for y in range(0, len(image)):
        for x in range(0, len(image[0])):
            if(image[y][x] == 0):
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
        minX = group[0][0]
        maxX = group[0][0] 
        minY = group[0][1] 
        maxY = group[0][1] 
       
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

    width = groupSize[2] - groupSize[0] + 3
    height = groupSize[3] - groupSize[1] + 3

    minX = groupSize[0]
    minY = groupSize[1]

    image = np.ones((height, width), np.uint8) * 255

    for pixel in group:
        x, y = pixel
        image[y - minY + 1][x - minX + 1] = 0

    return image

def getTips(img):
    window = np.ones((3, 3), dtype=np.uint8)
    tips = []

    for i in range(1, len(img[0])):  # x
        for j in range(2, len(img)): # y
            if(img[j][i] == 0):
                window = img[j-1:j+2, i-1:i+2]
                value = np.sum(np.logical_not(window))
                if(value == 2):
                    tips.append((i, j))
    return tips

def getCenter(img):
    window = np.ones((3, 3), dtype=np.uint8)
    center = (0,0)
    max = 0
    for i in range(1, len(img[0])):  # x
        for j in range(2, len(img)): # y
            if(img[j][i] == 0):
                window = img[j-1:j+2, i-1:i+2]
                value = np.sum(np.logical_not(window))
                if(value > max):
                    center = (i, j)
                    max = value
    return center

def getEccentricity(img):
    tips = getTips(img)
    center = getCenter(img)

    tipsDistance = []
    
    for tip in tips:
        tipsDistance.append(distance.euclidean(center, tip))
    r_max = np.max(tipsDistance)
    r_min = np.min(tipsDistance)
    e = (r_max - r_min) / (r_max + r_min)
    return e
    

def checkGroups(img, groups, groupSize):
    soloGears = []
    okGears = []
    gearsTeeths = []
    for index, group in enumerate(groups):
        subimg = createSubImage(group, groupSize[index])
        subimg = floodFill(subimg) # fechamento
        skeleton = skeletonize(cv2.bitwise_not(subimg))
        skeleton = cv2.bitwise_not(skeleton.astype('uint8') * 255)
        #show(subimg)
        #show(skeleton)
        e = getEccentricity(skeleton)
        if(e > 0.2):
            img = colorizeGroup(img, group, 240)
        else:
            soloGears.append(group)
            okGears.append(checkAngles(skeleton))
            gearsTeeths.append(getNumberOfTeeths(skeleton))

    return soloGears, okGears, gearsTeeths, img

def trimTips(skeleton):
    skeleton_copy = skeleton.copy()
    center = getCenter(skeleton)
    r = center[0]
    for x in range(0, len(skeleton[0])):
            for y in range(0, len(skeleton)):
                if(skeleton[y][x] == 0):
                    d =  math.sqrt((x - center[0])**2 + (y - center[1])**2)
                    if(d > r/2):
                        skeleton_copy[y][x] = 255

    return skeleton_copy

def getNumberOfTeeths(skeleton):
    skeleton_copy = trimTips(skeleton)
    tips = getTips(skeleton_copy)
    return len(tips)

def checkAngles(skeleton):
    skeleton_copy = trimTips(skeleton)
    tips = getTips(skeleton_copy)
    center = getCenter(skeleton_copy)
    ref_angle = 360/len(tips)
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
    print(ref_angle)
    print(angles)

    for i in range(0, len(angles)-1):
        if(angles[i+1] - angles[i] > 1.2 * ref_angle):
            return False
    if((360 + angles[0]) - angles[-1] > 1.2 * ref_angle):
        return False
    return True

def show(img):
    cv2.imshow('Resultado', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
         
# def vertexDistance(img):
# 	janela = np.ones((5, 5), dtype=np.uint8)
# 	max1_coords = []
# 	max2_coords = []
# 	distancias = []
# 	count_vizinhos = []
# 	for i in range(len(img[0])-4):
# 		for j in range(len(img)-4):
# 			if(img[j][i] == 0):             
# 					janela_pixels = img[i:i+5, j:j+5] 
# 					total_pixels_pretos = np.sum(np.logical_not(janela_pixels))
# 					coords = np.argwhere(janela_pixels == 0) #verifica se o pixel eh preto
#             else:
            
# 			# if len(coords) > 1:
# 			# 	coords = coords[np.argsort(coords[:, 0])]
# 			# 	max1_coords.append([i+coords[0, 0], j+coords[0, 1]])
# 			# 	max2_coords.append([i+coords[1, 0], j+coords[1, 1]])
# 			# 	distancia = distance.euclidean(max1_coords[-1], max2_coords[-1])

# 	# Cria a matriz resultante
# 	#distancia = np.column_stack((np.array(max1_coords), np.array(max2_coords), np.array(distancias)))
# 	return distancia

