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

# Reference: https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html

import argparse
import cv2
import numpy as np

def draw_contour(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Using circle detection function to get result.
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=2, minDist=30,
                               param1=100, param2=30,
                               minRadius=15, maxRadius=20)
    
    processed_image = image.copy()
    circle_center_image = np.zeros_like(image)
    
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            # cv2.circle(image, center, 1, (255, 255, 255), 3)  # Change color to white
            # circle outline
            radius = i[2]
            cv2.circle(processed_image, center, radius, (0, 255, 0), 3)  # Change color to white
            # circle center image
            cv2.circle(circle_center_image, center, 1, (255, 255, 255), 3)  # Change color to white
    
    cv2.destroyAllWindows() 
    cv2.imshow("image", image)
    cv2.imshow("processed_image", processed_image)
    cv2.imshow("circle_center_image", circle_center_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def count_rings(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Using circle detection function to get result.
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=2, minDist=30,
                               param1=100, param2=30,
                               minRadius=15, maxRadius=20)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        circle_num = len(circles[0, :])
    return f"There are {circle_num} coins in the image."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--index", type=str, default=None)
    args = parser.parse_args()
    
    if args.image and args.index:
        image = args.image
        index = args.index
        if index == "0":
            draw_contour(image)
        elif index == "1":
            count_rings(image)
        else:
            print("Please specify the index of the question")
    else:
        print("Please specify the image path")

if __name__ == "__main__":
    main()