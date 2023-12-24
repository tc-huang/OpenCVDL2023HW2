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

import gradio as gr
import numpy as np
import os
# import cv2

with gr.Blocks() as demo:
    w = h = 320
    gr.ImageEditor(
        #"/Users/tchuang/GitHub/OpenCVDL2023HW2/data/raw/Dataset_OpenCvDl_Hw2/Q1/coins.jpg"
        value={
            "background": None,
            "layers": ["/Users/tchuang/GitHub/OpenCVDL2023HW2/src/black.png"],
            "composite": None
            },
        brush=gr.Brush(
            colors=["#FFFFFF"],
            color_mode="fixed"
        ),
        eraser=False,
        # crop_size=(320, 320),
        image_mode="RGB",
        interactive=True,
        height=320,
        width=320,
        sources=(),
        # scale=10,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
    # save a rgb black 320x320 image
    # image = np.zeros((320, 320, 3), dtype=np.uint8)
    # cv2.imwrite("black.png", image)
    
