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
import argparse

def closing(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Binarize the grayscale image, assigning values of 0 or 255 only. (threshold = 127)
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # Pad the image with zeros based on the kernel size (K=3)
    thresh = cv2.copyMakeBorder(thresh, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Perform dilation on the image
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    # Perform erosion on the image
    erosion = cv2.erode(dilation, kernel, iterations=1)
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
    erosion = cv2.erode(thresh, kernel, iterations=1)
    # Perform dilation on the image
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    # Show the result
    cv2.imshow("Original", image)
    cv2.imshow("Opening", dilation)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    pass

if __name__ == "__main__":
    main()