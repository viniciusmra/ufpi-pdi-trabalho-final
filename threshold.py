from PIL import Image
import numpy

from collections import deque

#Binariza a imagem usando o método de Otsu
def binarize(filename):
    sourceImage = numpy.asarray(Image.open(filename).convert('L'))
    threshold = otsu(sourceImage)
    binary = Image.open(filename).convert('L').point(lambda x: 255 if x > threshold else 0)
    return binary

# Método de Otsu (So ta funcionando se a imagem tiver pelo menos 1 pixel totalmente preto e 1 totalmente branco)
def otsu(img):
    hist, bins = numpy.histogram(img, bins=256, range=(0, 256))
    bins = bins[:-1]
    w0 = numpy.cumsum(hist)
    w1 = numpy.cumsum(hist[::-1])[::-1]
    mu0 = numpy.cumsum(bins * hist) / w0
    mu1 = (numpy.cumsum((bins * hist)[::-1]) / w1[::-1])[::-1]
    sigma_b_squared = ((mu0 - mu1) ** 2) * w0 * w1
    idx = numpy.argmax(sigma_b_squared)
    return bins[idx]

#Retorna todos os vizinhos com a mesma cor de um pixel
def neighbors(img, pixel):
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

def checkBorder(img):
    groups = []
    for x in range(0, len(img)):
        if(img[x][0] == 0):
            groups.append(neighbors(img, (x, 0)))

        if(img[x][len(img[0])-1] == 0):
            groups.append(neighbors(img, (x, len(img[0])-1)))
    
    for y in range(0, len(img[0])):
        if(img[0][y] == 0):
            groups.append(neighbors(img, (0, y)))

        if(img[len(img)-1][y] == 0):
            groups.append(neighbors(img, (len(img)-1, y)))
    
    for group in groups:
        for pixel in group:
            x, y = pixel
            img[x][y] = 200

    return img
def paintGroup(img ,group, grayscale):
    for pixel in group:
            x, y = pixel
            img[x][y] = grayscale

def pixelsGroups(img):
    new_image = img.copy()
    groups = []
    
    for x in range(0, len(new_image)):
        for y in range(0, len(new_image[0])):
            if(new_image[x][y] == 0):
                groups.append(neighbors(new_image, (x, y)))
                paintGroup(new_image, groups[-1], 1)
    return groups

def groupSize(groups):
    sizes = []
    for group in groups:
        minX = group[0][0]
        maxX = group[0][0]
        minY = group[0][1]
        maxY = group[0][1]
        for pixel in group:
            if(pixel[0] > maxX):
                maxX = pixel[0]
            if(pixel[0] < minX):
                minX = pixel[0]
            if(pixel[1] > maxY):
                maxY = pixel[1]
            if(pixel[1] < minY):
                minY = pixel[1]
        sizes.append((maxX - minX, maxY - minY))
    return sizes

if __name__ == '__main__':
    #img = numpy.asarray(binarize('gear_hack.png'))
    img = numpy.asarray(binarize('engrenagens.png'))
    img_copy = img.copy()
    img_copy = checkBorder(img_copy)
    groups = pixelsGroups(img_copy)
    print(groupSize(groups))
    #for x in range(0, len(img_copy)):



    Image.fromarray(img_copy).show()
