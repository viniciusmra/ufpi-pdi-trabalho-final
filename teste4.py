import cv2
import numpy as np

def extract_components(image_path):
    # Carrega a imagem em escala de cinza
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.bitwise_not(img)
    # Aplica uma limiarização para binarizar a imagem
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Encontra os componentes conexos na imagem binarizada
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    # Cria uma lista para armazenar as coordenadas dos componentes
    components = []

    # Itera sobre todos os componentes (exceto o fundo)
    for i in range(1, num_labels):
        # Encontra as coordenadas (x, y) de todos os pixels pertencentes ao componente
        component_coords = np.column_stack(np.where(labels == i))
        
        # Adiciona a lista de componentes
        components.append(component_coords)

    # Retorna a lista de componentes
    return components

components = extract_components("images/engrenagens.png")

# Itera sobre todos os componentes e exibe as coordenadas
for i, component in enumerate(components):
    print(f"Componente {i+1}: {component}")