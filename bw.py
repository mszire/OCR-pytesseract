# import cv2
# im_gray = cv2.imread('processed_image.png', cv2.IMREAD_GRAYSCALE)
# cv2.imwrite('processed_image1.png', im_gray)


#Thrashhole 
# import cv2
# import numpy as np
# img = cv2.imread('enhanced.opencv_frame_0.png')
# retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)

# # grayscaled = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
# # retval2, threshold2 = cv2.threshold(grayscaled, 12, 255, cv2.THRESH_BINARY)
# gaus = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

# # cv2. imshow('original' , img)
# # cv2.imshow('threshold', threshold)
# # cv2.imshow('threshold2', threshold2)
# cv2.imshow('gaus', gaus) 
# cv2.imwrite('finally.png', gaus)
# cv2.waitKey (0)

# cv2. destroyAllWindows ()

# import cv2
# import numpy as np

# img = cv2.imread('opencv_frame_1.png')
# img = cv2.GaussianBlur(img,(5,5),0)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# mask = np.zeros((gray.shape),np.uint8)
# kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

# close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1)
# div = np.float32(gray)/(close)
# res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
# res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)
# # cv2.imwrite('finally.png', res2)

# thresh = cv2.adaptiveThreshold(res,255,0,1,19,2)
# contour,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# max_area = 0
# best_cnt = None
# for cnt in contour:
#     area = cv2.contourArea(cnt)
#     if area > 1000:
#         if area > max_area:
#             max_area = area
#             best_cnt = cnt

# cv2.drawContours(mask,[best_cnt],0,255,-1)
# cv2.drawContours(mask,[best_cnt],0,0,2)

# res = cv2.bitwise_and(res,mask)

# cv2.imwrite('finally.png', res)


import cv2
import numpy as np

def box_extraction(img_for_box_extraction_path, cropped_dir_path):

    img = cv2.imread(img_for_box_extraction_path, 0)  # Read the image
    (thresh, img_bin) = cv2.threshold(img, 128, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    img_bin = 255-img_bin  # Invert the image

    cv2.imwrite("Image_bin.jpg",img_bin)
   
    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//40
     
    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    cv2.imwrite("verticle_lines.jpg",verticle_lines_img)

# Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    cv2.imwrite("horizontal_lines.jpg",horizontal_lines_img)

# Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    # cv2.imwrite("img_final_bin.jpg",img_final_bin)
    # Find contours for image, which will detect all the boxes
    # im2, contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    im2,contours,hierachy=cv2.findContours(img_final_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    idx = 0
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)

# If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
        if (w > 80 and h > 20) and w > 3*h:
            idx += 1
            new_img = img[y:y+h, x:x+w]
            cv2.imwrite(cropped_dir_path+str(idx) + '.png', new_img)

box_extraction("ms5.png", "./Cropped/")