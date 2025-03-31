import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
import io
import matplotlib.pyplot as plt 

st.set_page_config(
    page_title="Image Classfier",
    page_icon="🇳🇵",
    layout="wide",
)

# Define your class dictionary
class_names = {'Donuts': 0, 'French Fries': 1, 'Fried Rice': 2, 'Samosa': 3}
class_names_rev = {v: k for k, v in class_names.items()}  # Reverse the dictionary for easy lookup

# Load your trained my_model
from model import CustomCNN  # Import your my_model architecture
my_model = CustomCNN(num_classes=4)
my_model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu'), weights_only=True))
my_model.eval()  # Set my_model to evaluation mode

# Define image transformations (same as during training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # Normalize for RGB images
])

# Streamlit UI
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
        # Predict Button
        if st.button("Predict"):
            predict_clicked = True  # Update flag when button is clicked

    # **Column 3: Prediction and Confidence Chart**
    with col3:
        if uploaded_file is not None and predict_clicked:
            # Transform and classify the image
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                output = my_model(image_tensor)
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
