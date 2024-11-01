from typing import Any
from typing import Dict
import gradio as gr
import sys
import io
import time
from gradio.themes.builder_app import themes
from patsy.state import scale
from pydantic import BaseModel
from pydantic import Extra
from pydantic import model_validator
from pdf2image import convert_from_path



class Interface(BaseModel):
    gr: Any = None
    interface: Any = None

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        """
        Validate that api key and python package exists in environment.

        This function checks if the `gradio` Python package is installed in the environment. If the package is not found, it raises a `ValueError`
        with an appropriate error message.

        Args:
            cls (object): The class to which this method belongs.
            values (Dict): A dictionary containing the environment values.
        Return:
            Dict: The updated `values` dictionary with the `gradio` package imported.
        Raise:
            ValueError: If the `gradio` package is not found in the environment.

        """

        try:
            import gradio as gr

            values["gr"] = gr
        except ImportError:
            raise ValueError(
                "Could not import gradio python package. "
                "Please install it with `pip install gradio`."
            )
        return values

    class Config:
        """Configuration for this pydantic object."""

        #extra = Extra.forbid
        arbitrary_types_allowed = True

    def prepare_interface(
            self,
            upload_file,
            serve_file,
            process_query,
            share=False,
    ):
        def welcome(name):
            return f"Welcome to Causal Copilot, {name}!"

        js = """
        function createGradioAnimation() {
            var container = document.createElement('div');
            container.id = 'gradio-animation';
            container.style.fontSize = '2em';
            container.style.fontWeight = 'bold';
            container.style.textAlign = 'center';
            container.style.marginBottom = '20px';

            var text = 'Welcome to Causal Copilot!';
            for (var i = 0; i < text.length; i++) {
                (function(i){
                    setTimeout(function(){
                        var letter = document.createElement('span');
                        letter.style.opacity = '0';
                        letter.style.transition = 'opacity 0.5s';
                        letter.innerText = text[i];

                        container.appendChild(letter);

                        setTimeout(function() {
                            letter.style.opacity = '1';
                        }, 50);
                    }, i * 250);
                })(i);
            }

            var gradioContainer = document.querySelector('.gradio-container');
            gradioContainer.insertBefore(container, gradioContainer.firstChild);

            return 'Animation created';
        }
        """

        with self.gr.Blocks(js=js, theme = gr.themes.Soft() ) as demo:
            with self.gr.Row():
                chatbot = self.gr.Chatbot(type="messages", bubble_full_width=False, height=500)

            with self.gr.Row():
                msg = self.gr.Textbox(
                    scale=3,
                    label="Question",
                    info="Please describe your causal discovery problem here and press enter.",
                )

                btn_upload = self.gr.UploadButton(
                        label="""
                        📁 
                        Upload your data file (.csv)
                        """,
                        scale=0,
                        file_types=["file"],
                        file_count="single"
                )

                msg.submit(
                    process_query,
                    [msg],
                    [chatbot]
                )

                msg.submit(process_query, inputs=msg, outputs=chatbot)

                # Upload Button for Dataset
                btn_upload.upload(
                    upload_file, btn_upload, queue=False
                )

                # Download Report
                download_trigger = gr.Button(value = """
                        📁
                        Download your report (.pdf)
                        """)

                download_trigger.click(
                    serve_file, outputs=gr.File(label=""))


        demo.queue().launch(share=share)
        self.interface = demo

    def close(self):
        """
        Close the Gradio interface.

        This method closes the Gradio interface associated with the chatbot.
        It calls the `close` method of the interface object stored in the `self.interface` attribute.

        Args:
            self (object): The instance of the class.
        Return:
            None
        """

        self.interface.close()
