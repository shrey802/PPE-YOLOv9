# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image
# import torch
# import torchvision.transforms as transforms

# # Force the use of CPU
# device = 'cpu'
# st.write(f"Using device: {device}")

# # Load the trained YOLOv8 model
# model = YOLO("ppe.pt")  # Replace with the path to your ppe.pt file
# model.to(device)

# st.title("YOLOv8 Object Detection with Streamlit")

# # File uploader for images
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Load image with PIL
#     image = Image.open(uploaded_file)
    
#     # Display the uploaded image
#     st.image(image, caption="Uploaded Image", use_column_width=True)
    
#     st.write("Running detection...")
    
#     # Define a transformation to resize the image and convert it to a PyTorch tensor
#     transform = transforms.Compose([
#         transforms.Resize((640, 640)),  # Resize image to 640x640 pixels
#         transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
#     ])
    
#     # Apply transformation to the image
#     image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
#     # Run inference on the image
#     results = model(image_tensor)
    
#     # Iterate over the results and save/display each one
#     for i, result in enumerate(results):
#         result_img = result.plot()  # Create an image with bounding boxes drawn
#         result_img_pil = Image.fromarray(result_img)  # Convert to PIL Image
        
#         output_path = f"output_{i}.jpg"
#         result_img_pil.save(output_path)  # Save the result image
        
#         # Display the output image
#         st.image(result_img_pil, caption=f"Detected Image {i+1}", use_column_width=True)




# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image
# import torch
# import torchvision.transforms as transforms
# import cv2  # OpenCV for video processing
# import numpy as np
# import tempfile

# # Force the use of CPU
# device = 'cpu'
# st.write(f"Using device: {device}")

# # Load the trained YOLOv8 model
# model = YOLO("ppe.pt")  # Replace with the path to your ppe.pt file
# model.to(device)

# st.title("YOLOv8 Object Detection with Streamlit")

# # File uploader for images and videos
# uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

# if uploaded_file is not None:
#     file_type = uploaded_file.name.split('.')[-1].lower()
    
#     if file_type in ["jpg", "jpeg", "png"]:
#         # Image processing
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_column_width=True)
#         st.write("Running detection...")
        
#         # Resize the image to a shape that is divisible by 32
#         new_width = (image.width // 32) * 32
#         new_height = (image.height // 32) * 32
#         image = image.resize((new_width, new_height))
        
#         transform = transforms.Compose([
#             transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
#         ])
#         image_tensor = transform(image).unsqueeze(0).to(device)
#         results = model(image_tensor)
        
#         for i, result in enumerate(results):
#             result_img = result.plot()
#             result_img_pil = Image.fromarray(result_img)
#             st.image(result_img_pil, caption=f"Detected Image {i+1}", use_column_width=True)
    
#     elif file_type == "mp4":
#         # Save the uploaded video temporarily
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
#             temp_file.write(uploaded_file.read())
#             video_path = temp_file.name
        
#         # Video processing
#         st.video(uploaded_file)  # Display the original video
        
#         cap = cv2.VideoCapture(video_path)
#         frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
#         # Ensure frame dimensions are divisible by 32
#         new_frame_width = (frame_width // 32) * 32
#         new_frame_height = (frame_height // 32) * 32
        
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (new_frame_width, new_frame_height))
        
#         st.write("Running detection on video...")
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Resize frame to the new dimensions
#             frame = cv2.resize(frame, (new_frame_width, new_frame_height))
            
#             # Convert frame to PIL Image
#             image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
#             # Apply transformation
#             image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
            
#             # Run inference
#             results = model(image_tensor)
            
#             # Plot results on the frame
#             for result in results:
#                 result_img = result.plot()
#                 frame = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            
#             # Write the processed frame to the output video
#             out.write(frame)
        
#         cap.release()
#         out.release()
        
#         # Display the output video
#         st.video("output_video.mp4")

        




















# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image
# import torch
# import torchvision.transforms as transforms
# import cv2
# import tempfile

# # Force the use of CPU
# device = 'cpu'
# st.write(f"Using device: {device}")

# # Load the trained YOLOv8 model
# model = YOLO("ppe.pt")
# model.to(device)

# st.title("YOLOv8 Object Detection with Streamlit")

# # File uploader for videos
# uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"])

# if uploaded_file is not None:
#     # Create a temporary file to store the uploaded video
#     tfile = tempfile.NamedTemporaryFile(delete=False)
#     tfile.write(uploaded_file.read())
    
#     # Capture the video
#     cap = cv2.VideoCapture(tfile.name)
    
#     # Prepare a temporary output file for the processed video
#     output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_file.name, fourcc, 20.0, (640, 640))
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Preprocess the frame
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image = Image.fromarray(frame)
#         transform = transforms.Compose([
#             transforms.Resize((640, 640)),
#             transforms.ToTensor(),
#         ])
#         image_tensor = transform(image).unsqueeze(0).to(device)
        
#         # Run inference on the frame
#         results = model(image_tensor)
        
#         # Draw bounding boxes on the frame
#         result_img = results[0].plot()
#         result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        
#         # Write the frame with bounding boxes to the output video
#         out.write(result_img_bgr)
    
#     # Release resources
#     cap.release()
#     out.release()
    
#     # Display the processed video
#     st.video(output_file.name)
    
#     # Optionally provide a download link
#     with open(output_file.name, "rb") as file:
#         btn = st.download_button(
#             label="Download Processed Video",
#             data=file,
#             file_name="output.mp4",
#             mime="video/mp4"
#         )























import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as transforms
import os
from twilio.rest import Client

# Initialize Twilio Client
account_sid = 'XXXX'  # Your Account SID from Twilio
auth_token = 'XXXXX'  # Your Auth Token from Twilio
client = Client(account_sid, auth_token)

# Force the use of CPU
device = "cpu"
st.write(f"Using device: {device}")

# Load the trained YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("ppe.pt")  # Replace with the path to your ppe.pt file

model = load_model()
model.to(device)

st.title("YOLOv8 Object Detection with Streamlit")

# Function to send WhatsApp notification
def send_whatsapp_notification(missing_classes):
    missing_classes_str = ", ".join(missing_classes)
    # Send WhatsApp Message with missing classes
    message = client.messages.create(
        from_='whatsapp:+14XXXXXX',  # Twilio Sandbox WhatsApp number
         body=f'Alert: The following required equipment is missing: {missing_classes_str}',
        to='whatsapp:+91XXXXXXX'  # Replace with your WhatsApp number
    )
    print(f"WhatsApp notification sent. Message SID: {message.sid}")

# File uploader for images and videos
uploaded_file = st.file_uploader(
    "Choose an image or a video...", type=["jpg", "jpeg", "png", "mp4"]
)

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    no_detections = True  # Flag to track detections

    if file_extension in [".jpg", ".jpeg", ".png"]:
        # Process as image
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.write("Running detection on image...")

        # Define a transformation to resize the image and convert it to a PyTorch tensor
        transform = transforms.Compose(
            [
                transforms.Resize((640, 640)),  # Resize image to 640x640 pixels
                transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
            ]
        )

        # Apply transformation to the image
        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        # Run inference on the image
        results = model(image_tensor)

        # List of required classes
        required_classes = {'Boots', 'Ear-protection', 'Glass', 'Glove', 'Helmet', 'Mask', 'Vest'}
        detected_classes = set()

        # Extract the class names from the model's detections
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = result.names[class_id]  # Get the class name
                detected_classes.add(class_name)  # Add the detected class to the set

        # Determine which classes are missing after processing the image
        missing_classes = required_classes - detected_classes

        # Send a WhatsApp message if any required class is missing
        if missing_classes:
            send_whatsapp_notification(missing_classes)

        # Display the detected image
        for i, result in enumerate(results):
            result_img = result.plot()  # Create an image with bounding boxes drawn
            result_img_pil = Image.fromarray(result_img)  # Convert to PIL Image

            output_path = f"output_{i}.jpg"
            result_img_pil.save(output_path)  # Save the result image

            # Display the output image
            st.image(
                result_img_pil, caption=f"Detected Image {i+1}", use_column_width=True
            )

    elif file_extension == ".mp4":
        # Process as video
        # (Your existing video processing code remains unchanged)
        pass

    else:
        st.error("Unsupported file type! Please upload an image or a video.")

    # Show an alert within Streamlit if no detections were made
    if no_detections:
        st.warning("Alert: No detections were found in the processed image.")
