import streamlit as st
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from model import build_model
from PIL import Image
import os
import time

# Define class names
class_names = ['barbell biceps curl', 'push up']

# Set the computation device
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMAGE_RESIZE = 256

# Validation transforms
def get_test_transform(image_size):
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return test_transform

@st.cache_data()
def load_model():
    # Load the pre-trained model
    weights_path = "../outputs/best_model.pth"  # Update with your model path
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    model = build_model(fine_tune=False, num_classes=len(class_names)).to(DEVICE).eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def preprocess_frame(frame):
    transform = get_test_transform(IMAGE_RESIZE)
    img = Image.fromarray(frame).convert('RGB')
    img = transform(img).unsqueeze(0)
    return img.to(DEVICE)

def classify_frame(frame, model):
    outputs = model(frame)
    predictions = torch.softmax(outputs, dim=1).cpu().detach().numpy()
    predicted_class = np.argmax(predictions[0])
    return class_names[predicted_class]

def main():
    st.title("Workout Video Classifier")

    uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(temp_video_path)

        if st.button("Process"):
            placeholder = st.empty()
            placeholder.write("Processing...")  # Display "Processing..." message

            model = load_model()
            vidcap = cv2.VideoCapture(temp_video_path)

            frame_count = 0
            total_fps = 0
            predicted_classes = []

            while True:
                ret, frame = vidcap.read()
                if not ret:
                    break

                image_tensor = preprocess_frame(frame)
                start_time = time.time()
                predicted_class = classify_frame(image_tensor, model)
                end_time = time.time()

                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1

                predicted_classes.append(predicted_class)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Calculate and display average FPS
            avg_fps = total_fps / frame_count
            # st.write(f"Average FPS: {avg_fps:.2f}")

            # Display the most frequent predicted class
            most_common_class = max(set(predicted_classes), key=predicted_classes.count)
            placeholder.write(f"Prediction: {most_common_class}")  # Update placeholder with prediction

            vidcap.release()
            cv2.destroyAllWindows()

            # Remove the temporary video file
            os.remove(temp_video_path)

if __name__ == "__main__":
    main()
