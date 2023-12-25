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
import glob
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def show_images():
    inference_dataset_path = Path(__file__).parent.parent / "inference_dataset"
    print(inference_dataset_path)
    cat_dir_path = f"{inference_dataset_path}/Cat"
    dog_dir_path = f"{inference_dataset_path}/Dog"
    cat_image_paths = glob.glob(f"{cat_dir_path}/*")
    dog_image_paths = glob.glob(f"{dog_dir_path}/*")
    cat_image = Image.open(cat_image_paths[0]).resize((224, 224))
    dog_image = Image.open(dog_image_paths[0]).resize((224, 224))
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(cat_image)
    plt.title("Cat")
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(dog_image)
    plt.title("Dog")
    plt.show()

def show_comparasion():
    figure_path = Path(__file__).parent / "accuracy_comparision.png"
    print(figure_path)
    figure = Image.open(figure_path)
    figure.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, default=None)
    args = parser.parse_args()
    
    if args.index:
        index = args.index
        if index == "1":
            show_images()
        elif index == "3":
            show_comparasion()
        else:
            print("Please specify the index of the question")
    else:
        print("Please specify the index of the question")

if __name__ == "__main__":
    main()