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
import timm
import torchsummary

def show_model_structure():
    NUM_FINETUNE_CLASSES = 
    model = timm.create_model(
        "vgg19_bn",
        pretrained=False,
        checkpoint_path="/Users/tchuang/GitHub/OpenCVDL2023HW2/src/solutions/q4/results/checkpoint/best_model_28_accuracy=0.9398.pt",
        num_classes=10
    )
    torchsummary.summary(model, (3, 32, 32), device="cpu")

def show_accuracy_and_loss():
    figure_path = "" 