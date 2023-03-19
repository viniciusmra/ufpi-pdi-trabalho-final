import cv2
import numpy as np

# Carrega a imagem em escala de cinza
imagem = cv2.imread("1gear_nohole.png", 0)
imagem = cv2.bitwise_not(imagem)
# Aplica threshold na imagem
_, thresh = cv2.threshold(imagem, 127, 255, cv2.THRESH_BINARY)
height, width = thresh.shape
size = height * width

# Aplica transformação morfológica para reduzir a imagem ao esqueleto
skel = np.zeros(thresh.shape, np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
done = False
while not done:
    eroded = cv2.erode(thresh, element)
    temp = cv2.dilate(eroded, element)
    temp = cv2.subtract(thresh, temp)
    skel = cv2.bitwise_or(skel, temp)
    thresh = eroded.copy()
    zeros = size - cv2.countNonZero(thresh)
    if zeros == size:
        done = True

# Exibe o esqueleto
cv2.imshow("Esqueleto", skel)
cv2.waitKey(0)
cv2.destroyAllWindows()