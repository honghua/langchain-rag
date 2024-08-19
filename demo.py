import gradio as gr
import os
import shutil
from pathlib import Path
from datastore import DataStoreGenerator
from query_processor import QueryProcessor

class RAGUI:
    def __init__(self):
        self.data_path = Path("data/uploaded_docs")
        self.data_path.mkdir(exist_ok=True)
        self.query_processor = None

    def upload_file(self, files):
        file_paths = []
        for file in files:
            if file is not None:
                dest_path = os.path.join(self.data_path, os.path.basename(file.name))
                shutil.move(file.name, dest_path)
                file_paths.append(dest_path)
        
        if file_paths:
            return f"Files uploaded successfully: {', '.join(file_paths)}"
        return "No files uploaded."

    def process_uploads(self):
        DataStoreGenerator.generate_data_store(self.data_path, overwrite=True)
        self.query_processor = QueryProcessor()
        return "Documents processed and data store created."

    def chat(self, message, history):
        if self.query_processor is None:
            return "Please process the uploaded documents first."
        
        response = self.query_processor.process_query(message)
        # Return the new message and updated history
        return "", history + [(message, response)]

    def launch(self, share=False):
        with gr.Blocks() as demo:
            gr.Markdown("# RAG Demo")
            
            with gr.Tab("Upload Documents"):
                upload_button = gr.File(label="Upload Files", file_count="multiple")
                text_output = gr.Textbox()
                process_button = gr.Button("Process Uploaded Documents")

                upload_button.upload(self.upload_file, upload_button, text_output)
                process_button.click(self.process_uploads, outputs=text_output)

            with gr.Tab("Chat"):
                chatbot = gr.Chatbot()
                msg = gr.Textbox()
                clear = gr.Button("Clear")

                msg.submit(self.chat, [msg, chatbot], [msg, chatbot])
                clear.click(lambda: None, None, chatbot, queue=False)

        demo.launch(share=share)

if __name__ == "__main__":
    ui = RAGUI()
    ui.launch(share=True)