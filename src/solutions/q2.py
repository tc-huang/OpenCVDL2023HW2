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
    # cv2.imshow("Original image", gray)
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # part 1
    equalized = cv2.equalizeHist(gray)
    # plot image and frequency by matplotlib
    # cv2.imshow("Equalized with OpenCV", equalized)
    hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0, 256])
    
    
    # part 2
    # caculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # caculate pdf
    pdf = hist / sum(hist)
    # caculate cdf
    cdf = pdf.cumsum()
    # caculate mapping
    mapping = [round(255 * i) for i in cdf]
    # mapping
    new_image = gray.copy()
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            new_image[i, j] = mapping[gray[i, j]]
    
    # plot image and frequency by matplotlib
    # cv2.imshow("Equalized with my function", new_image)
    hist_new_image = cv2.calcHist([new_image], [0], None, [256], [0, 256])
    
    # plot 3 image and 3 bar chart
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(gray, cmap="gray")
    axs[0, 0].set_title("Original image")
    axs[1, 0].bar(range(256), hist_gray[:, 0])
    axs[1, 0].set_title("Histogram of Original")

    axs[0, 1].imshow(equalized, cmap="gray")
    axs[0, 1].set_title("Equalized with OpenCV")
    axs[1, 1].bar(range(256), hist_equalized[:, 0])
    axs[1, 1].set_title("Histogram of Equalized (OpenCV)")

    axs[0, 2].imshow(new_image, cmap="gray")
    axs[0, 2].set_title("Equalized Manually")
    axs[1, 2].bar(range(256), hist_new_image[:, 0])
    axs[1, 2].set_title("Histogram of Equalized (Manually)")
    
    plt.show()
def main():
    pass

if __name__ == "__main__":
    main()