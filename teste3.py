import cv2
import numpy as np

# Criar uma matriz com 4 pixels brancos
img = np.zeros((2,2), dtype=np.uint8)
img.fill(255)
img[0][1] = 0
print(img)

# Salvar a matriz como uma imagem
cv2.imwrite('imagem.png', img)