import streamlit as st
import docx
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

def read_docx(path):
    """Reads and extracts text from a .docx file."""
    doc = docx.Document(path)
    return '\n'.join(paragraph.text for paragraph in doc.paragraphs)

def model_initialization():
    """Initializes the Qwen model and processor."""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    return model, processor

# Streamlit App
st.title("Chatbot with Qwen Model")
st.write("This app uses a Qwen Vision-Language model for processing .docx files and generating responses.")

# File upload
uploaded_file = st.file_uploader("Upload a .docx file", type="docx")

if uploaded_file is not None:
    st.write("File uploaded successfully!")

    # Read and display content of the file
    doc_content = read_docx(uploaded_file)
    st.subheader("Extracted Content:")
    st.text_area("File Content", doc_content, height=300)

    # Initialize model
    with st.spinner("Initializing model..."):
        model, processor = model_initialization()
    st.success("Model loaded successfully!")

    # User input for interaction
    user_input = st.text_input("Enter your query:", "What is the content about?")

    if st.button("Generate Response"):
        with st.spinner("Processing..."):
            inputs = processor(text=[user_input], return_tensors="pt")
            outputs = model.generate(**inputs)
            response = processor.decode(outputs[0], skip_special_tokens=True)
        st.subheader("Model Response:")
        with st.expander("See Full Output"):
            st.write(response)

