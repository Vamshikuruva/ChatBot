# DocuBot - A Document based Question Answering ChatBot.


This project demonstrates an AI-powered chatbot built using the Qwen Vision-Language model. The chatbot processes `.docx` files, extracts content, and provides intelligent, context-aware responses through a user-friendly Streamlit interface.

## Features

- **Document Processing**: Extracts and processes text from uploaded `.docx` files.
- **Interactive Chatbot**: Responds to user queries based on document content.
- **Vision-Language Model**: Leverages the Qwen2-VL-7B-Instruct model for advanced text understanding.
- **Scalable Interface**: Built with Streamlit for seamless user interaction.
- **Adaptive Fine-Tuning**: Supports domain-specific customizations for enhanced accuracy.

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - [Transformers](https://huggingface.co/docs/transformers)
  - [Torch](https://pytorch.org/)
  - [Streamlit](https://streamlit.io/)
  - [python-docx](https://python-docx.readthedocs.io/)
- **Platform**: Local and Cloud Deployment

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Vamshikuruva/DocuBot.git
   cd DocuBot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Upload a `.docx` file through the interface.
2. Enter a query in the text box.
3. View the chatbot's response based on the document content.

## Project Structure

- **`app.py`**: Main Streamlit application file.
- **`requirements.txt`**: List of dependencies.
- **`README.md`**: Project documentation.

## Future Enhancements

- Support for additional document formats (e.g., PDF).
- Multi-modal query handling (text + images).
- Integration with cloud services for large-scale deployment.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing the Qwen Vision-Language model.
- [Streamlit](https://streamlit.io/) for enabling rapid application development.
