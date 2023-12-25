# Copyright 2023 tc-haung
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
import argparse


def dilate(gray):
    # structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    height, width = gray.shape
    dilation = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            dilation[i, j] = np.max(gray[i:i+3, j:j+3])
    return dilation
            

def erode(gray):
    # structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    height, width = gray.shape
    erosion = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            erosion[i, j] = np.min(gray[i:i+3, j:j+3])
    return erosion

def closing(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Binarize the grayscale image, assigning values of 0 or 255 only. (threshold = 127)
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # Pad the image with zeros based on the kernel size (K=3)
    thresh = cv2.copyMakeBorder(thresh, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    # Perform dilation on the image
    dilation = dilate(thresh)
    # Perform erosion on the image
    erosion = erode(dilation)
    # Show the result
    cv2.imshow("Original", image)
    cv2.imshow("Closing", erosion)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def opening(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Binarize the grayscale image, assigning values of 0 or 255 only. (threshold = 127)
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # Pad the image with zeros based on the kernel size (K=3)
    thresh = cv2.copyMakeBorder(thresh, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Perform erosion on the image
    erosion = erode(thresh)
    # Perform dilation on the image
    dilation = dilate(erosion)
    # Show the result
    cv2.destroyAllWindows()
    cv2.imshow("Original", image)
    cv2.imshow("Opening", dilation)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--index", type=str, default=None)
    args = parser.parse_args()
    
    if args.image and args.index:
        image = args.image
        index = args.index
        if index == "0":
            closing(image)
        elif index == "1":
            opening(image)
        else:
            print("Please specify the index of the question")
    else:
        print("Please specify the image path")

if __name__ == "__main__":
    main()