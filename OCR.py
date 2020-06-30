# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:58:06 2020
 
@author: Shubham Darak
"""
 
#import the necessary package
import cv2
import numpy as np
import argparse
import time
import math
import matplotlib.pyplot as plt
import skimage.io as io
import os
from imutils.object_detection import non_max_suppression
import pytesseract
 
def decode_predictions(scores, geometry):
  # grab the number of rows and columns from the scores volume, then
  # initialize our set of bounding box rectangles and corresponding
  # confidence scores**
  (numRows, numCols) = scores.shape[2:4]
  boxes = []
  confidences = []
 
  # loop over the number of rows**
  for y in range(0, numRows):
    # extract the scores (probabilities), followed by the geometrical
    # data used to derive potential bounding box coordinates that
    # surround text**
    scoresData = scores[0, 0, y]
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]
 
    # loop over the number of columns**
    for x in range(0, numCols):
      # if our score does not have sufficient probability, ignore it**
      if scoresData[x] < 0.25:
        continue
      # compute the offset factor as our resulting feature maps will
      # be 4x smaller than the input image**
      (offsetX, offsetY) = (x * 4.0, y * 4.0)
      # extract the rotation angle for the prediction and then
      #   compute the sin and cosine**
      angle = anglesData[x]
      cos = np.cos(angle)
      sin = np.sin(angle)
 
      # use the geometry volume to derive the width and height of
      # the bounding box**
      h = xData0[x] + xData2[x]
      w = xData1[x] + xData3[x]
 
      # compute the rotated rect for
      # the text prediction bounding box**
      offset = (offsetX + (cos * xData1[x]) + (sin * xData2[x]), offsetY - (sin * xData1[x]) + (cos * xData2[x]))
      p1 = (-sin * h + offset[0], -cos * h + offset[1])
      p3 = (-cos * w + offset[0],  sin * w + offset[1])
      center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
 
      # add the bounding box coordinates and probability score to
      # our respective lists**
      boxes.append((center, (w,h), -angle * 180.0 / math.pi))
      confidences.append(float(scoresData[x]))
  return (boxes, confidences)
  
def text_detection(image_path):  
  # load the input image and grab the image dimensions**
  image = cv2.imread(image_path)
  orig = image.copy()
  (H, W) = image.shape[:2]
  # set the new width and height and then determine the ratio in change
  # for both the width and height**
  (newW, newH) = (640,640)
  rW = W / float(newW)
  rH = H / float(newH)
 
  # resize the image and grab the new image dimensions**
  image = cv2.resize(image, (newW, newH))
  (H, W) = image.shape[:2]
 
  # define the two output layer names for the EAST detector model that
  # we are interested -- the first is the output probabilities and the
  # second can be used to derive the bounding box coordinates of text**
  layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
 
  # load the pre-trained EAST text detector**
  print("[INFO] loading EAST text detector...")
  net = cv2.dnn.readNet("/content/frozen_east_text_detection.pb")
 
  # construct a blob from the image and then perform a forward pass of
  # the model to obtain the two output layer sets**
  blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),(123.68, 116.78, 103.94), swapRB=True, crop=False)
  start = time.time()
  net.setInput(blob)
  (scores, geometry) = net.forward(layerNames)
  end = time.time()
 
 
  (boxes, confidences) = decode_predictions(scores, geometry)
 
  # apply non-maxima suppression to suppress weak, overlapping bounding boxes**
  indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, 0.25, 0.24)
 
  results = []
  box = []
  
  for i in indices:
    # get 4 corners of the rotated rect**
    vertices = cv2.boxPoints(boxes[i[0]])
 
    #scale the bounding box coordinates based on the respective ratios**
    pt1 = int(vertices[0][0] * rW - 10)
    pt2 = int(vertices[0][1] * rH + 10)
    pt3 = int(vertices[1][0] * rW - 10 )
    pt4 = int(vertices[1][1] * rH - 10)
    pt5 = int(vertices[2][0] * rW + 10)
    pt6 = int(vertices[2][1] * rH - 10)
    pt7 = int(vertices[3][0] * rW + 10)
    pt8 = int(vertices[3][1] * rH + 10)
  
    pts = np.array([[pt1,pt2],[pt3,pt4],[pt5,pt6],[pt7,pt8]],np.int32)
    pts=pts.reshape((-1,1,2))
    # draw the bounding box on the image
    cv2.polylines(orig,[pts], True,(0, 255, 255),3)
 
  plt.imshow(orig)
 
  image = cv2.imread(image_path)
  orig = image.copy()
  import pytesseract
  image = cv2.imread(image_path)
  orig = image.copy()
  # initialize the list of results
  results = []
  # loop over the bounding boxes
  for i in indices:
    # get 4 corners of the rotated rect**
    vertices = cv2.boxPoints(boxes[i[0]])
 
    # scale the bounding box coordinates based on the respective ratios**
    pt1 = int(vertices[0][0] * rW - 10)
    pt2 = int(vertices[0][1] * rH + 10)
    pt3 = int(vertices[1][0] * rW - 10 )
    pt4 = int(vertices[1][1] * rH - 10)
    pt5 = int(vertices[2][0] * rW + 10)
    pt6 = int(vertices[2][1] * rH - 10)
    pt7 = int(vertices[3][0] * rW + 10)
    pt8 = int(vertices[3][1] * rH + 10)
  
    pts = np.array([[pt1,pt2],[pt3,pt4],[pt5,pt6],[pt7,pt8]],np.int32)
    pts=pts.reshape((-1,1,2)) 
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = orig[y:y+h, x:x+w].copy()
 
    ## (2) make mask
    pts = pts - pts.min(axis=0)
 
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
 
    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)
 
    rect = cv2.minAreaRect(pts)
 
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    width = int(rect[1][0])
    height = int(rect[1][1])
 
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],[0, 0],[width-1, 0],[width-1, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(dst, M, (width, height))
    sharpen_kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharpen = cv2.filter2D(warped, -1, sharpen_kernel)
    sharpen_kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    img = cv2.filter2D(warped, -1, sharpen_kernel)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
 
    #roi = img
    #plt.imshow(img)
    #plt.imshow(img)
    roi = img
    
    # in order to apply Tesseract v4 to OCR text we must supply
    # (1) a language, (2) an OEM flag of 4, indicating that the we
    # wish to use the LSTM neural net model for OCR, and finally
    # (3) an OEM value, in this case, 7 which implies that we are
    # treating the ROI as a single line of text
    config = ("-l eng --oem 1 --psm 7")
    text = pytesseract.image_to_string(roi, config=config)
    # add the bounding box coordinates and OCR'd text to the list
    # of results
    results.append(text)
  # Read image using opencv
  # loop over the results
  for (text) in results:
    # display the text OCR'd by Tesseract
      print("OCR TEXT")
      print("========")
      print("{}\n".format(text))
  return orig