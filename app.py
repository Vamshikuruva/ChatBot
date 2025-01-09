import streamlit as st
import docx
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def read_docx(path):
    """Reads and extracts text from a .docx file."""
    try:
        doc = docx.Document(path)
        return '\n'.join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as e:
        return f"Error Reading File: {e}"

@st.cache_resource
def model_initialization():
    """Initializes the Qwen model and processor."""
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype='auto',
            device_map='auto',
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        return model, processor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise

def get_output(model, processor, content, question):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": content},
                    {"type": "text", "text": question},
                ],
            }
        ]

        # Prepare inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Generate output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0] if output_text else "No response generated."
    except Exception as e:
        st.error(f"Error during inference: {e}")
        return "An error occurred during processing."


# Streamlit App
st.set_page_config(layout="wide") 

st.title("DocuBot")
st.write("This app uses a Qwen Vision-Language model for processing .docx files and generating responses.")

# Check if a document has been uploaded
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

if not st.session_state.uploaded:
    # File upload
    uploaded_file = st.file_uploader("Upload a .docx file", type="docx")
    if uploaded_file is not None:
        st.session_state.uploaded = True
        st.session_state.doc_content = read_docx(uploaded_file)
else:
    st.write("Document uploaded successfully! Reload the page to upload a new document.")

# After upload, display content and chat interface
if st.session_state.uploaded:
        #splitting views
        cols=st.columns(2)

        #left col
        with cols[0]:
            # Read and display content of the file
            st.subheader("Extracted Content:")
            st.text_area("File Content", st.session_state.doc_content, height=500)
        #right col
        with cols[1]:
            # Initialize model
            try:
                with st.spinner("Initializing model..."):
                    model, processor = model_initialization()
                    st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading the model: {e}")
                st.stop()
            
            # Chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            # User input for interaction
            user_input = st.text_input("Your Message:", placeholder="Ask something about the document...")

            if st.button("Send"):
                if not user_input.strip():
                    st.warning("Please enter a valid query.")
                    
                if user_input.strip():
                    with st.spinner("Processing..."):
                        try:
                            response = get_output(model, processor,st.session_state.doc_content, user_input.strip())
                            
                            # Update chat history
                            st.session_state.chat_history.append({"user": user_input, "bot": response})
                        except Exception as e:
                            st.error(f"Error generating response: {e}")
            # Display chat history
            for chat in st.session_state.chat_history:
                st.markdown(f"**You:** {chat['user']}")
                st.markdown(f"**Bot:** {chat['bot']}")
                st.divider()  # Add a visual separator between messages
