from skimage.morphology import skeletonize, skeletonize_3d
import cv2

# Carrega a imagem em escala de cinza
imagem = cv2.imread("grayscaletest.png", 0)
imagem = cv2.bitwise_not(imagem)


# Binariza a imagem
ret, thresh = cv2.threshold(imagem, 127, 255, cv2.THRESH_BINARY)
# cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Gera um esqueleto 2D conexo usando o algoritmo de Zhang-Suen
skeleton = skeletonize(thresh)
teste = cv2.bitwise_not(skeleton.astype('uint8') * 255)

# Exibe a imagem do esqueleto
#cv2.imshow('Esqueleto', teste )

#imagem = cv2.imread("caminho_da_imagem.jpg")
cv2.imwrite("imagem_salva3.png", teste)
#cv2.waitKey(0)
#cv2.destroyAllWindows() 
