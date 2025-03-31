import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
import matplotlib.pyplot as plt
from torchvision import models

st.set_page_config(
    page_title="Image Classifier",
    page_icon="🇳🇵",
    layout="wide",
)

# Define class dictionary
class_names = {'Donuts': 0, 'French Fries': 1, 'Fried Rice': 2, 'Samosa': 3}
class_names_rev = {v: k for k, v in class_names.items()}  # Reverse dictionary for lookup

# Load trained model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 4)  # Adjust output layer
    model.load_state_dict(torch.load("transfer_learning.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("Food Classifier")
st.write("Upload a Food image to get a prediction")

# Create 3 columns
col1, col2, col3 = st.columns([1, 1, 1])  # Equal width for each column

# **Column 1: Upload Image**
with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# **Column 2: Display Image**
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Ensure RGB format
    if image.mode != "RGB":
        image = image.convert("RGB")

    predict_clicked = False  # Flag to track button click

    with col2:
        st.header("Preview")
        new_width = 400
        new_height = 300
        # Resize the image using PIL
        resized_image = image.resize((new_width, new_height))
        
        # Display the resized image
        st.image(resized_image, caption="Uploaded Image", use_column_width=False)
        #st.image(image, caption="Uploaded Image", use_column_width=True)
        # Predict Button
        if st.button("Predict"):
            predict_clicked = True  # Update flag when button is clicked

    # **Column 3: Prediction and Confidence Chart**
    with col3:
        if uploaded_file is not None and predict_clicked:
            # Transform and classify the image
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1).squeeze()
                _, predicted_class = torch.max(output, 1)

            # Display predicted class
            predicted_class_name = class_names_rev[predicted_class.item()]
            st.markdown(f"<h3 style='color: #4CAF50;'>Predicted: {predicted_class_name}</h3>", unsafe_allow_html=True)

            # Create confidence bar chart
            fig, ax = plt.subplots()
            ax.bar(list(class_names.keys()), probabilities.tolist(), color="skyblue")
            ax.set_xlabel("Classes")
            ax.set_ylabel("Confidence")
            st.pyplot(fig)
