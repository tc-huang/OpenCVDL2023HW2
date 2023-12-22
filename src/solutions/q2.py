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

import argparse
import cv2
import matplotlib.pyplot as plt

def histogram_equalization(image_path):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # part 1
    equalized = cv2.equalizeHist(gray)
    # plot image and frequency by matplotlib
    cv2.imshow("Original image", gray)
    cv2.imshow("Equalized with OpenCV", equalized)
    
    
    # part 2
    # caculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0, 256])
    # processe image
    hist = hist.flatten()
    hist_equalized = hist_equalized.flatten()
    # TODO:

def main():
    pass

if __name__ == "__main__":
    main()