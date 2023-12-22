import subprocess

import gradio as gr
import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
from PIL import Image
import numpy as np


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
        subprocess.run(["python", "src/solutions/q1.py", "--image", image, "--index", "1"])
    else:
        print("Please load a image")


# # Q2
# def q2_1_preprocessing(video):
#     if video:
#         video = video.name
#         print("Call 2.1 Preprocessing...")
#         subprocess.run(
#             ["python", "src/solutions/q2.py", "--index", "0", "--video", video]
#         )
#     else:
#         print("Please load a video")


# def q2_2_video_tracking(video):
#     if video:
#         video = video.name
#         print("Call 2.2 Video tracking...")
#         subprocess.run(
#             ["python", "src/solutions/q2.py", "--index", "1", "--video", video]
#         )
#     else:
#         print("Please load a video")


# # Q3
# def q3_dimension_reduction(image):
#     pass
#     if image:
#         image = image.name
#         print("Call 3. Dimension reduction...")
#         subprocess.run(["python", "src/solutions/q3.py", "--image", image])
#     else:
#         print("Please load an image")


# # Q4
# def q4_1_show_model_structure(load_image_1):
#     # if load_image_1:
#     #     load_image_1 = load_image_1.name
#     #     print("Call 4.1 Keypoints...")
#     #     subprocess.run(["python", "./src/q4/q4_1_keypoints.py", "--image1", load_image_1])
#     # else:
#     #     print("Please load image 1")
#     pass


# def q4_2_show_accuracy_and_loss(load_image_1, load_image_2):
#     # if load_image_1 and load_image_2:
#     #     load_image_1 = load_image_1.name
#     #     load_image_2 = load_image_2.name
#     #     print("Call 4.2 Matched Keypoints...")
#     #     subprocess.run(["python", "./src/q4/q4_2_matched_keypoints.py", "--image1", load_image_1, "--image2", load_image_2])
#     # else:
#     #     print("Please load image 1 and image 2")
#     pass


# def q4_3_predict():
#     pass


# def q4_4_reset():
#     pass


# # Q5
# def q5_1_show_images():
#     print("Call 5.1 Show Augmented Images...")
#     subprocess.run(["python", "src/q5/q5_1_show_augmented_images.py"])


# def q5_2_show_model_structure():
#     print("Call 5.2 Show Model Structure...")
#     subprocess.run(["python", "src/q5/q5_2_show_vgg19bn_structure.py"])


# def q5_3_show_comparasion():
#     print("Call 5.3 Show Acc and Loss...")
#     subprocess.run(["python", "src/q5/q5_3_show_acc_and_loss.py"])


# def inference(load_image):
#     print("5.4 Inference")
#     # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     # transform = transforms.Compose([
#     #     transforms.ToTensor(),
#     #     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
#     # ])
#     # print("Load model...")
#     # model_path = "src/q5/model/vgg16_model.pth"
#     # model = torchvision.models.vgg19_bn(num_classes=10)
#     # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#     # model.to(device)
#     # model.eval()
#     # print("Model loaded")

#     # image = Image.open(load_image)
#     # image_tensor = transform(image)
#     # print("Inference...")
#     # with torch.no_grad():
#     #     output = model(image_tensor.unsqueeze(0))

#     # softmax = nn.Softmax(dim=1)
#     # probs = softmax(output)
#     # labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#     # print("Inference done")

#     # return list(probs[0].numpy()), f"Predicted = {labels[torch.argmax(probs[0]).item()]}"
#     pass


# def q5_4_inference(load_image):
#     print("Call 5.4 Inference...")
#     if load_image:
#         pass
#     else:
#         print("Please load image")


with gr.Blocks() as demo:
    with gr.Row():
        # Input
        with gr.Column():
            image = gr.File(label="Load Image")
            video = gr.File(label="Load Video")
        # Q1
        with gr.Group():
            gr.Markdown("""## 1. Background Subtraction""")
            gr.Button("1. Background Subtraction").click(
                fn=q1_background_substraction, inputs=video, outputs=None
            )

    #     # Q2
    #     with gr.Group():
    #         gr.Markdown("""## 2. Optical Flow""")
    #         gr.Button("2.1 Preprocessing").click(
    #             q2_1_preprocessing, inputs=video, outputs=None
    #         )
    #         gr.Button("2.2 Video tracking").click(
    #             q2_2_video_tracking, inputs=video, outputs=None
    #         )

    # with gr.Row():
    #     # Q3
    #     with gr.Group():
    #         gr.Markdown("""## 3. PCA""")
    #         gr.Button("3. Dimension Reduction").click(
    #             fn=q3_dimension_reduction, inputs=image, outputs=None
    #         )

    #     # Q4
    #     with gr.Group():
    #         gr.Markdown("""## 4. MNIST Classifier Using VGG19""")
    #         gr.Button("1. Show Model Structure").click(
    #             fn=q4_1_show_model_structure, inputs=None, outputs=None
    #         )
    #         gr.Button("2. Show Accuracy and Loss").click(
    #             fn=q4_2_show_accuracy_and_loss, inputs=None, outputs=None
    #         )
    #         gr.Button("3. Predict").click(fn=q4_3_predict, inputs=None, outputs=None)
    #         gr.Button("4. Reset").click(fn=q4_4_reset, inputs=None, outputs=None)
    #     with gr.Group():
    #         im = gr.ImageEditor(
    #             # value = { 
    #             #     "background": None,
    #             #     "layers" : [],  
    #             #     "composite": np.zeros((320, 320, 3), dtype=np.uint8)
    #             # },
    #             type="numpy",
    #             image_mode="RGB",
    #             # sources=(),
    #             brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"),
    #             crop_size="1:1",
    #             label="draw number",
    #             height = 320,
    #             width=320,
    #             eraser=False,
    #             interactive=False
    #             # brush=False
    #         )
    #         # gr.Sketchpad()
    # # Q5
    # with gr.Row():
    #     with gr.Group():
    #         gr.Markdown("## 5. ResNet50")
    #         q5_image = gr.File(label="Load Image")

    #         gr.Button("5.1 Show Images").click(
    #             q5_1_show_images, inputs=None, outputs=None
    #         )
    #         gr.Button("5.2 Show Model Structure").click(
    #             q5_2_show_model_structure, inputs=None, outputs=None
    #         )
    #         gr.Button("5.3 Show Comparasion").click(
    #             q5_3_show_comparasion, inputs=None, outputs=None
    #         )
    #         gr.Button("5.4 Inference").click(q5_4_inference, inputs=None, outputs=None)
    #     with gr.Group():
    #         inference_image = gr.Image()
    #         predicted_class_label = gr.Textbox(label="Predicted Class Label")
    #             # inference_button.click(q5_4_predicted_class, inputs=load_image2, outputs=[inference_image, predicted_class_label])
    #             # inference_button = gr.Button("5.4 Inference")


    # demo.queue()
    demo.launch(server_name="0.0.0.0")
