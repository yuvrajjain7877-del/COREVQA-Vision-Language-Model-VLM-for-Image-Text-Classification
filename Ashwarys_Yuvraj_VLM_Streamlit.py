### Importing all the Necessary Libraries--------------------------------------------------------------------

import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor







### Setting up Page Definition--------------------------------------------------------------------------------------------
st.set_page_config(
    page_title= "COREVQA",
    page_icon = "🖼️",
    layout = "centered"
)





### Loading the Backend Model------------------------------------------------------------- 
classifier = joblib.load("SVM_VLM.pkl")








### Loading the Pretrained Model--------------------------------------------------------------------------------
@st.cache_resource
def load_clip():
    
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    model.eval()
    return model, processor

model, processor = load_clip()    
    







### Frontend GUI-------------------------------------------------------------------------------------------
st.title("Visual Question Answering Experiment")

st.subheader("Image Based Question Answering System")

st.write(
    "Upload an Image and ask a Question about it."
    "The model will predict whether the statement is true or false"
)








## Image and Question Uploading-----------------------------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an Image",
    type = ["jpg", "jpeg", "png"]
)

Question = st.text_input(
    "Enter Your Statement", 
    placeholder = "Example: Although there are 3 people in the image, no  one is holding a camera and one person in the image is holding a jacket in his right hand."
)

image = None
if uploaded_file is not None:
    
    image = Image.open(uploaded_file).convert("RGB")
    
    st.image(
        image,
        caption = "Uploaded Image",
        use_container_width = True
    )











### Connecting the Data with Model and generate Embeddings-------------------------------------------------------------------------------------------    
if st.button("Generate Results :"):
    
    if image is None:
        st.warning("Please Upload an Image")
        
    elif not Question.strip():
        st.warning("Please enter a Question.")
        
    else:
        with st.spinner("Generating Results :"):
            
            inputs = processor(
                text = [Question],
                images = [image],
                return_tensors = "pt",
                padding = True,
                Truncation = True
            )
            
            with torch.no_grad():
                
                outputs = model(**inputs)
                
                image_embeddings = outputs.image_embeds
                text_embeddings = outputs.text_embeds
                
            
            combined_embeddings = torch.cat(
                [image_embeddings, text_embeddings], dim = 1
            )
            
            features = combined_embeddings.cpu().numpy()
            
            prediction = classifier.predict(features)
            
            predicted_answer = bool(prediction[0])
            
            if prediction[0] == 1:
                predicted_answer = True
            else:
                predicted_answer = False
            
            st.subheader("Predicted Result")
            st.write("Predicted Answer : ", predicted_answer)