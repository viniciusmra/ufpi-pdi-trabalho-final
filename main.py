from PIL import Image
import cv2

import defs

if __name__ == '__main__':
    image = cv2.imread("images/1gear.png", 0)
    binary_image = defs.binarize(image)
    
    noborder_image = defs.clearBorder(binary_image, 255)

    nohole_image = defs.floodFill(noborder_image)

    groups = defs.getGroups(nohole_image)
    
    groupSize = defs.getBounds(groups)

    subimg = defs.createSubImage(groups[0], groupSize[0])

    cv2.imshow('Esqueleto', cv2.resize(subimg, (800, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # #img = numpy.asarray(binarize('gear_hack.png'))
    # img = numpy.asarray(defs.binarize('engrenagens.png'))
    # print(img)
    # img_copy = img.copy()
    # img_copy = defs.clearBorder(img_copy, 200)
    # groups = defs.pixelsGroups(img_copy)
    # print(defs.groupSize(groups))
    # #for x in range(0, len(img_copy)):



    # Image.fromarray(img_copy).show()