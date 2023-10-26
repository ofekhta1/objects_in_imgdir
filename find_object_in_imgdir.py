import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import shutil
import tkinter as tk
from tkinter import filedialog
def create_output_directory(image_file,path):
    image_name = os.path.splitext(image_file)[0]
    if(path!=images_dir+"\\archive"):
        output_dir = os.path.join(path, image_name)
    else:
        output_dir=path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir
def create_combined_file():
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    detected_object = template[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    detected_object_resized = cv2.resize(detected_object, (image.shape[1], image.shape[0]))
    combined_image = np.hstack((image, detected_object_resized))
    output_dir = create_output_directory(image_files, images_dir + r"\archive")
    combined_image_path = os.path.join(output_dir, "combined_" + image_files)
    cv2.imwrite(combined_image_path, combined_image)
root = tk.Tk()
root.withdraw()
image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
template_path = filedialog.askopenfilename(title="Select the Template")
template = cv2.imread(template_path)
new_images_dir = input("Enter a new value for images_dir(the def is C:\python\images_objects ): ")
images_dir = new_images_dir if new_images_dir else r"C:\python\images_objects"
for image_files in os.listdir(images_dir):
    image_path = os.path.join(images_dir, image_files)
    if os.path.isfile(image_path) and any(image_files.lower().endswith(ext) for ext in image_extensions):
     image_path = os.path.join(images_dir, image_files)
     image = cv2.imread(image_path)
     if image is None or template is None:
      print("Error: Unable to load one or both images.")
      exit()
     image_height, image_width, _ = image.shape
     template = cv2.resize(template, (image_width, image_height))
     template = template.astype(image.dtype)
     result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
     threshold = 0.2  
     if max_val >= threshold:
       h, w, _ = template.shape
       top_left = max_loc
       bottom_right = (top_left[0] + w, top_left[1] + h)
       cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Green rectangle for the found object
       print("The object (e.g., tattoo) exists in both images!")
       create_combined_file()
      
    #    if(os.path.exists(image_path)):
    #     shutil.copy(image_path, matched_video_path)
     else:
       MIN_MATCH_COUNT = 18
       img1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # queryImage
       img2 = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE) # trainImage
       sift = cv2.SIFT_create()
       kp1, des1 = sift.detectAndCompute(img1,None)
       kp2, des2 = sift.detectAndCompute(img2,None)
       FLANN_INDEX_KDTREE = 1
       index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
       search_params = dict(checks = 50)
       flann = cv2.FlannBasedMatcher(index_params, search_params)
       matches = flann.knnMatch(des1,des2,k=2)
       good = []
       for m,n in matches:
        if m.distance < 0.7*n.distance:
         good.append(m)
       if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
       else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
       draw_params = dict(matchColor = (0,255,0), # draw matches in green color
       singlePointColor = None,
       matchesMask = matchesMask, # draw only inliers
       flags = 2)
       img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
       if len(good) > MIN_MATCH_COUNT:
        print("The object (e.g., tattoo) exists in both images!")
        #plt.imshow(img3, 'gray')
        #plt.show(block=True)
        create_combined_file()
       else:
     #plt.imshow(img3, 'gray')
     #plt.show(block=True)
         print("The object (e.g., tattoo) does NOT exist in both images or the similarity is too low.")

    
