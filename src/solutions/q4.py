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

from PIL import Image
from pathlib import Path
import argparse
import matplotlib.pyplot as plt


def show_loss_and_acc():
    figure_path = Path(__file__).parent / "loss_and_acc.png"
    figure = Image.open(figure_path)
    figure.show()

def plot_probabilities():
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, default=None)
    parser.add_argument('--probs', help='List of floating-point numbers')
    args = parser.parse_args()
    
    if args.index:
        index = args.index
        if index == "2":
            show_loss_and_acc()
        elif index == "3":
            if args.probs:
                probs = [float(num_str) for num_str in args.probs.split()]
                labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
                plt.figure()
                plt.bar(labels, probs)
                plt.ylabel('Probability')
                plt.xlabel('Class')
                plt.title('Probability of each class')
                plt.show()
                print("Done")
            plot_probabilities()
        else:
            print("Please specify the index of the question")
    else:
        print("Please specify the index of the question")

if __name__ == "__main__":
    main()