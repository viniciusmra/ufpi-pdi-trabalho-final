from PIL import Image
import cv2
from skimage.morphology import skeletonize, skeletonize_3d

import defs

if __name__ == '__main__':
    image = cv2.imread("images/teste15.png", 0)
    binary_image = defs.binarize(image)
    print("binarizou")
    
    noborder_image = defs.clearBorder(binary_image, 240)
    print("limpou a borda")

    #nohole_image = defs.floodFill(noborder_image)
    defs.show(noborder_image)
    print("Antes da segmentação")

    groups = defs.getGroups(noborder_image)
    
    groupSize = defs.getBounds(groups)

    print("Antes da excentricidade")
    
    soloGears, okGears, gearsTeeths, noborder_image  = defs.checkGroups(noborder_image, groups, groupSize)
    print(okGears)
    for i, group in enumerate(soloGears):
        if(not okGears[i]):
            noborder_image = defs.colorizeGroup(noborder_image, group, 100)
    





    
        

    # skeleton = skeletonize(cv2.bitwise_not(subimg))
    # skeleton = cv2.bitwise_not(skeleton.astype('uint8') * 255)
    # print(defs.getEccentricity(skeleton))
    #skeleton_binary = defs.binarize(skeleton)
    
    #cv2.imshow('Resultado', cv2.resize(noborder_image, (1000, 500)))
    cv2.imshow('Resultado', noborder_image)
    #cv2.imwrite('resultado3.png', noborder_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #distance = defs.ellementoestruturanteflamengoganhoudovasco(skeleton) #testar
    #print(distance)
    