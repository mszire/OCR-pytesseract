from PIL import Image, ImageEnhance 
import cv2
im = Image.open("pi.png")
enhancer = ImageEnhance.Sharpness(im)
enhanced_im = enhancer.enhance(16)

enhanced_im.save("done1.png")


# import cv2
# import numpy as np
# img = cv2.imread('enhanced.opencv_frame_0.png',0)
# equ = cv2.equalizeHist(img)
# res = np.hstack((img,equ)) #stacking images side-by-side
# cv2.imwrite('res.png',res)

# img = cv2.imread('res.png') #load rgb image
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert it to hsv

# for x in range(0, len(hsv)):
#     for y in range(0, len(hsv[0])):
#         hsv[:,:,2] += value

# img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# cv2.imwrite("image_processed.jpg", img)

# from PIL import Image, ImageEnhance 
# im = Image.open("enhanced.opencv_frame_0.png")
# enhancer = ImageEnhance.Brightness(im)
# enhanced_im = enhancer.enhance(1.6)
# enhanced_im.save("enhanced.sample5.png")
