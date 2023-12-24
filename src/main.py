import subprocess

import gradio as gr
import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import timm
import torch
import torch.nn as nn
import torchsummary
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, v2
from transformers import ConvNextImageProcessor, ResNetForImageClassification


# Q1
def q1_1_draw_contour(image):
    if image:
        image = image.name
        print("Call 1.1 Draw contour...")
        subprocess.run(["python", "src/solutions/q1.py", "--image", image, "--index", "0"])
    else:
        print("Please load a image")

def q1_2_count_rings(image):
    if image:
        image = image.name
        print("Call 1.2 Count Rings..")
        
        image = cv2.imread(image)
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
    else:
        print("Please load a image")

# Q2
def q2_histogram_equalization(image):
    if image:
        image = image.name
        print("Call 2. Histogram Equalization...")
        subprocess.run(
            ["python", "src/solutions/q2.py", "--image", image]
        )
    else:
        print("Please load a image")

# Q3
def q3_1_closing(image):
    if image:
        image = image.name
        print("Call 3.1 Closing...")
        subprocess.run(["python", "src/solutions/q3.py", "--image", image, "--index", "0"])
    else:
        print("Please load an image")

def q3_1_opening(image):
    if image:
        image = image.name
        print("Call 3.2 Opening...")
        subprocess.run(["python", "src/solutions/q3.py", "--image", image, "--index", "1"])
    else:
        print("Please load an image")

# Q4
model_vgg19bn = timm.create_model(
        "vgg19_bn",
        # pretrained=False,
        checkpoint_path="/Users/tchuang/GitHub/OpenCVDL2023HW2/src/solutions/q4/results/checkpoint/best_model_28_accuracy=0.9398.pt",
        num_classes=10
    )
data_transform = Compose(
    [
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Resize((32, 32)),
        v2.Lambda(lambda x: x.repeat(3, 1, 1)),
    ]
)

def q4_1_show_model_structure():
    print("Call 4.1 Show the Structure of VGG19 with BN...")
    torchsummary.summary(model_vgg19bn, (3, 32, 32), device="cpu")

def q4_2_show_accuracy_and_loss():
    print("Call Show Training/Validating Accuracy and Loss...")
    subprocess.run(["python", "./src/q4.py", "--index", "2"])


def q4_3_predict(image):
    print("Call 4.3 Predict...")
    image = image['composite']
    cv2.imwrite("composit.png", image)
    image = data_transform(image).unsqueeze(0)
    print(image.shape)
    with torch.no_grad():
        logits = model_vgg19bn(image)
    prediction = torch.softmax(logits, dim=1)
    print(prediction)
    result = torch.argmax(prediction, dim=1).item()
    return f"{result}"

def q4_4_reset():
    print("Call 4.4 Reset...")
    draw.clear()


# Q5
image_processor = ConvNextImageProcessor.from_pretrained("microsoft/resnet-50")
check_point = "/Users/tchuang/GitHub/OpenCVDL2023HW2/src/solutions/q5/results/detr-resnet-50_2023-12-23_17:11:43/checkpoint-2535"
model = ResNetForImageClassification.from_pretrained(
    check_point,
    ignore_mismatched_sizes=True,
    num_labels=1
)
model.classifier.add_module("2", nn.Sigmoid())

def q5_1_show_images():
    print("Call 5.1 Load the dataset and resize images...")
    subprocess.run(["python", "src/solutions/q5.py", "--index", "1"])


def q5_2_show_model_structure():
    print("Call 5.2 Show Model Structure...")
    torchsummary.summary(model, (3, 224, 224), device="cpu", depth=10)

def q5_3_show_comparasion():
    print("Call 5.3 Show Acc and Loss...")
    subprocess.run(["python", "src/solutions/q5.py", "--index", "3"])

def q5_4_inference(image): 
    if image:
        image = image.name
        image = Image.open(image).convert("RGB")
        print("Call 5.4  Use the beNer-trained model to run inference and show the predicted class label...")
        inputs = image_processor(image, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        prediction = np.round(logits).item()
        id2label = {0: "Cat", 1: "Dog"}
        print(f"Prediction: {id2label[prediction]}")
        return image, f"Prediction: {id2label[prediction]}"
    else:
        print("Please load a image")


with gr.Blocks() as demo:
    with gr.Row():
        # Input
        with gr.Column():
            image = gr.File(label="Load Image")
        # Q1
        with gr.Group():
            gr.Markdown("""## 1. Hough Circle Transform""")
            gr.Button("1.1 Draw Contour").click(
                fn=q1_1_draw_contour, inputs=image, outputs=None
            )
            bt = gr.Button("1.2 Count Rings")
            rings_num = gr.Textbox(label="")
            bt.click(
                fn=q1_2_count_rings, inputs=image, outputs=rings_num
            )

        # Q2
        with gr.Group():
            gr.Markdown("""## 2. Histogram Equalization""")
            gr.Button("2. Histogram Equalization").click(
                q2_histogram_equalization, inputs=image, outputs=None
            )

    with gr.Row():
        # Q3
        with gr.Group():
            gr.Markdown("""## 3. Morphology Operation """)
            gr.Button("3.1 Closing").click(
                fn=q3_1_closing, inputs=image, outputs=None
            )
            gr.Button("3.1 Opening").click(
                fn=q3_1_opening, inputs=image, outputs=None
            )

        # Q4
        with gr.Group():
            gr.Markdown("""## 4. MNIST Classifier Using VGG19""")
            gr.Button("1. Show Model Structure").click(
                fn=q4_1_show_model_structure, inputs=None, outputs=None
            )
            gr.Button("2. Show Accuracy and Loss").click(
                fn=q4_2_show_accuracy_and_loss, inputs=None, outputs=None
            )
            gr.Button("3. Predict").click(fn=q4_3_predict, inputs=None, outputs=None)
            gr.Button("4. Reset").click(fn=q4_4_reset, inputs=None, outputs=None)
        
        with gr.Group():
            draw = gr.Sketchpad(
                brush=gr.Brush(
                    colors=["#FFFFFF"],
                    color_mode="fixed"
                ),
                crop_size="1:1",
                eraser=False,
                image_mode='L'
            )
            
            io = gr.Interface(fn=q4_3_predict,inputs=draw, outputs="text",live=True)

    # Q5
    with gr.Row():
        with gr.Group():
            gr.Markdown("## 5. ResNet50")
            q5_image = gr.File(label="Load Image")
            with gr.Row():
                with gr.Column():
                    gr.Button("5.1 Show Images").click(
                        q5_1_show_images, inputs=None, outputs=None
                    )
                    gr.Button("5.2 Show Model Structure").click(
                        q5_2_show_model_structure, inputs=None, outputs=None
                    )
                    gr.Button("5.3 Show Comparasion").click(
                        q5_3_show_comparasion, inputs=None, outputs=None
                    )
                    inference_button = gr.Button("5.4 Inference")
                with gr.Column():
                    inference_image = gr.Image()
                    predicted_class_label = gr.Textbox(label="")
                    inference_button.click(q5_4_inference, inputs=q5_image, outputs=[inference_image, predicted_class_label])

    demo.launch(server_name="0.0.0.0")
