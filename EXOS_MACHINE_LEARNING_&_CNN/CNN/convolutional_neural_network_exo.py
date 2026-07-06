import cv2
import numpy as np

def apply_convolution():
    # 1. Créer une image de test simple (un carré blanc sur fond noir)
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
    
    # 2. Définir un filtre de détection de contours verticaux (Filtre Sobel)
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    
    # 3. Appliquer la convolution
    # cv2.filter2D fait glisser le kernel sur l'image
    output = cv2.filter2D(img, -1, kernel)
    
    # 4. Sauvegarder les images pour voir le résultat
    cv2.imwrite('/app/output/original.png', img)
    cv2.imwrite('/app/output/contour.png', output)
    print("Convolution terminée avec succès ! Regarde dans le dossier 'output'.")

if __name__ == "__main__":
    apply_convolution()