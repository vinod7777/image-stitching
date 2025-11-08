import numpy as np
import cv2
import glob
import os
import imutils

# Construct path to images relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path_pattern = os.path.join(script_dir, 'img', '*.jpg')
image_path = glob.glob(image_path_pattern)
images =[]

for image  in image_path:
    img = cv2.imread(image)
    images.append(img)
    cv2.imshow("Image",img)
    cv2.waitKey(0)

print("Attempting to stitch images...")
imageStitcher = cv2.Stitcher.create()
error , stitched_img = imageStitcher.stitch(images)

if not error:



    stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    cv2.imshow("Thresholded Image", thresh_img)
    cv2.waitKey(0)

    contours  = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    areas = max(contours, key=cv2.contourArea)

    mask =  np.zeros(thresh_img.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(areas)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    min_rect = thresh_img.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 0:
        min_rect = cv2.erode(min_rect, None)
        sub = cv2.subtract(min_rect, thresh_img)

    contours = cv2.findContours(min_rect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    areas = max(contours, key=cv2.contourArea)
    cv2.imshow("minRectangle image ", min_rect)
    cv2.waitKey(0)

    (x, y, w, h) = cv2.boundingRect(areas)
    stitched_img = stitched_img[y:y + h, x:x + w]

    # Crop 3% from each side
    print("Cropping 3% from each side...")
    h, w, _ = stitched_img.shape
    crop_top = int(h * 0.04)
    crop_bottom = int(h * 0.04)
    crop_left = int(w * 0.03)
    crop_right = int(w * 0.03)
    stitched_img = stitched_img[crop_top:h-crop_bottom, crop_left:w-crop_right]
    
    print("Stitching successful!")
    cv2.imwrite("stitched_img.png",stitched_img)
    cv2.imshow("Stitched Image",stitched_img)
    cv2.waitKey(0)

else:
    print("Stitching failed with error code:", error)
    print("Error codes:")
    print("1: ERR_NEED_MORE_IMGS - Not enough images to stitch.")
    print("2: ERR_HOMOGRAPHY_EST_FAIL - Homography estimation failed.")
    print("3: ERR_CAMERA_PARAMS_ADJUST_FAIL - Camera parameter adjustment failed.")