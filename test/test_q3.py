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

import src.solutions.q3 as q3

IMAGE_PATH_CLOSING = "data/raw/Dataset_OpenCvDl_Hw2/Q3/closing.png"
IMAGE_PATH_OPENING = "data/raw/Dataset_OpenCvDl_Hw2/Q3/opening.png"

def test_closing():
    q3.closing(IMAGE_PATH_CLOSING)

def test_opening():
    q3.opening(IMAGE_PATH_OPENING)