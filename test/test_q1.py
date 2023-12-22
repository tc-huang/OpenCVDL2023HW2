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

import src.solutions.q1 as q1

IMAGE_PATH = "data/raw/Dataset_OpenCvDl_Hw2/Q1/coins.jpg"

def test_draw_contour():
    q1.draw_contour(IMAGE_PATH)

def test_count_rings():
    q1.count_rings(IMAGE_PATH)