import cv2
from core import binarize, checkObjects, colorizeObject, getBorderObjects, getBounds, getObjects

if __name__ == '__main__':
    print("Binarizando a imagem...")
    rawImage = cv2.imread("images/teste15.png", 0)
    binaryImage = binarize(rawImage)
    
    print("Limpando a borda da imagem...")
    borderObjects = getBorderObjects(binaryImage)
    noBorderImage = binaryImage.copy()
    for object in borderObjects:
        noBorderImage = colorizeObject(noBorderImage, object, 255)
    
    print("Segmentando...")
    objects = getObjects(noBorderImage)
    objectBounds = getBounds(objects)

    print("Classificando engrenangens...")
    approvedGears, reprovedGears, undefinedGears, gearsTeeths  = checkObjects(objects, objectBounds)

    print("Gerando imagem final...")
    finalImamge = cv2.cvtColor(noBorderImage, cv2.COLOR_GRAY2RGB)
    for index, object in enumerate(objects):
        if approvedGears[index]:
            finalImamge = colorizeObject(finalImamge, object, [123, 177, 22])
        if reprovedGears[index]:
            finalImamge = colorizeObject(finalImamge, object, [130, 150, 231])
        if undefinedGears[index]:
            finalImamge = colorizeObject(finalImamge, object, [230, 230, 230])
    for object in borderObjects:
        finalImamge = colorizeObject(finalImamge, object, [230, 230, 230])
        

    cv2.imshow('Resultado', finalImamge)
    #cv2.imwrite('resultado5.png', finalImamge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    