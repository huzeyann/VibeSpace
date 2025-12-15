import logging
import os
import tempfile
import uuid
from typing import List, Union, Optional
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image
import numpy as np
import pandas as pd

from .vibe_blending import run_vibe_blend_safe, run_vibe_blend_not_safe
from .ipadapter_model import create_image_grid
from .feedback_viewer import create_feedback_viewer_tab, store_feedback_to_hf_dataset

# Hugging Face Datasets for feedback storage
try:
    from datasets import Dataset, load_dataset  # type: ignore
    from huggingface_hub import login  # type: ignore
    HF_DATASETS_AVAILABLE = True
except ImportError:
    Dataset = None  # type: ignore
    load_dataset = None  # type: ignore
    login = None  # type: ignore
    HF_DATASETS_AVAILABLE = False
    logging.warning("Hugging Face datasets not available. Feedback will not be stored.")

USE_HUGGINGFACE_ZEROGPU = os.getenv("USE_HUGGINGFACE_ZEROGPU", "false").lower() == "false" #"true"
DEFAULT_CONFIG_PATH = "./config.yaml"
# Hugging Face Dataset repository for storing feedback
# Set this to your Hugging Face username/dataset-name, e.g., "your-username/vibe-blending-feedback"
HF_FEEDBACK_DATASET_REPO = os.getenv("HF_FEEDBACK_DATASET_REPO", None)
HF_TOKEN = os.getenv("HF_TOKEN", None)

if USE_HUGGINGFACE_ZEROGPU:
    try:
        import spaces
    except ImportError:
        USE_HUGGINGFACE_ZEROGPU = False
        logging.warning("HuggingFace Spaces not available, running without GPU acceleration")

if USE_HUGGINGFACE_ZEROGPU:
    run_vibe_blend_safe = spaces.GPU(duration=60)(run_vibe_blend_safe)
    run_vibe_blend_not_safe = spaces.GPU(duration=60)(run_vibe_blend_not_safe)

    try:
        from .download_models import download_ipadapter
        download_ipadapter()
    except ImportError:
        logging.warning("Could not import download_models")


def create_gif_from_images(images: List[Image.Image], fps: float = 3.0) -> str:
    """
    Create a GIF from a list of PIL Images.
    
    Args:
        images: List of PIL Images to combine into a GIF
        fps: Frames per second for the GIF (default: 3.0)
    
    Returns:
        Path to the temporary GIF file
    """
    if not images:
        return None
    
    # Calculate duration in milliseconds (1000ms / fps)
    duration_ms = int(1000 / fps)
    
    # Create a temporary file for the GIF
    gif_path = os.path.join(tempfile.gettempdir(), f"vibe_blend_{uuid.uuid4().hex}.gif")
    
    # Save as GIF with loop
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0  # 0 = infinite loop
    )
    
    return gif_path


def load_gradio_images_helper(pil_images: Union[List, Image.Image, str]) -> List[Image.Image]:
    """
    Convert various image input formats to a list of PIL Images.
    """
    if pil_images is None:
        return []
    
    # Handle single image
    if isinstance(pil_images, np.ndarray):
        return Image.fromarray(pil_images).convert("RGB")
    if isinstance(pil_images, Image.Image):
        return pil_images.convert("RGB")
    if isinstance(pil_images, str):
        return Image.open(pil_images).convert("RGB")
    
    # Handle list of images
    processed_images = []
    for image in pil_images:
        if isinstance(image, tuple):  # Gradio gallery format
            image = image[0]
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, Image.Image):
            pass  # Already PIL Image
        else:
            continue
        processed_images.append(image.convert("RGB"))
    
    return processed_images




def create_vibe_blending_tab():
    """Create the vibe blending tab interface."""
    with gr.Tab("Vibe Blending"):
        gr.Markdown("""
        ## Vibe Blending Demo
        
        This is the demo for the paper "*Vibe Spaces for Creatively Connecting and Expressing Visual Concepts*".
        
        [Paper]() | [Code]() | [Website]()
        
        Given a pair of images, vibe blending will generate a set of images that creatively connect the input images.
        
        """)
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        gr.Markdown("**Step 1:** Upload 2 images")
                    with gr.Row():
                        input1 = gr.Image(label="Input 1", show_label=True, format="png")
                        input2 = gr.Image(label="Input 2", show_label=True, format="png")

                with gr.Accordion("Options", open=False):
                    with gr.Group():
                        with gr.Row():
                            alpha_start = gr.Slider(minimum=0, maximum=2, step=0.1, value=0.0, label="Start α", info="interpolation weight")
                            alpha_end = gr.Slider(minimum=0, maximum=2, step=0.1, value=1.0, label="End α", info="use α>1 for extrapolation")
                        # n_steps = gr.Slider(minimum=1, maximum=40, step=1, value=10, label="Number of Output Images")
                        n_steps = gr.Number(value=12, label="Number of Output Images", interactive=True)
                        with gr.Row():
                            extra_images = gr.Gallery(label="Extra Images (optional)", show_label=True, columns=3, rows=2, height=150)
                            negative_images = gr.Gallery(label="Negative Images (optional)", show_label=True, columns=3, rows=2, height=150)
                with gr.Group():
                    gr.Markdown("**Step 3:** Submit your feedback")
                    rating = gr.Radio(label="How do you like the results?", choices=["1", "2", "3", "4", "5"])
                    feedback_form = gr.TextArea(label="Feedback (optional)", show_label=True, lines=1, placeholder="Enter your feedback here...")
                    make_public = gr.Checkbox(label="Make feedback public", value=True, info="Allow this feedback to be visible in the public feedback viewer")
                    feedback_button = gr.Button("Submit Feedback", variant="secondary", size="sm")
                
            with gr.Column():
                with gr.Group():
                    gr.Markdown("**Step 2:** Run Vibe Blending")
                    blending_results = gr.Gallery(label="Gallery View", show_label=False, columns=4, rows=3, interactive=False)
                    with gr.Accordion("Vibe Blending Results (grid view)", open=False):
                        blending_results_grid = gr.Image(label="Grid View", show_label=True, format="png", interactive=False)
                    with gr.Accordion("Vibe Blending Results (GIF view)", open=False):
                        blending_results_gif = gr.Image(label="GIF View", show_label=True, format="gif", interactive=False)
                    blend_button = gr.Button("🔴 Run Vibe Blending", variant="primary")
        
        def _process_input_images(input1, input2, extra_images, negative_images):
            input1 = load_gradio_images_helper(input1)
            input2 = load_gradio_images_helper(input2)
            extra_images = load_gradio_images_helper(extra_images)
            negative_images = load_gradio_images_helper(negative_images)

            if extra_images is None:
                extra_images = []
            elif isinstance(extra_images, Image.Image):
                extra_images = [extra_images]

            if negative_images is None:
                negative_images = []
            elif isinstance(negative_images, Image.Image):
                negative_images = [negative_images]

            return input1, input2, extra_images, negative_images
        
        # Training wrapper function
        def blend_button_click(input1, input2, extra_images, negative_images, alpha_start, alpha_end, n_steps):
            input1, input2, extra_images, negative_images = _process_input_images(input1, input2, extra_images, negative_images)

            alpha_weights = np.linspace(alpha_start, alpha_end, n_steps+2)[1:-1].tolist()
            blended_images_list = run_vibe_blend_not_safe(input1, input2, extra_images, negative_images, DEFAULT_CONFIG_PATH, alpha_weights)
            blended_images_grid = create_image_grid(blended_images_list, rows=np.ceil(len(blended_images_list)/4).astype(int), cols=4)
            
            # Create GIF at 3 frames per second
            gif_path = create_gif_from_images(blended_images_list, fps=3.0)
            
            return blended_images_grid, blended_images_list, gif_path  # Return grid, list, and GIF for display
        
        blend_button.click(blend_button_click, inputs=[input1, input2, extra_images, negative_images, alpha_start, alpha_end, n_steps], outputs=[blending_results_grid, blending_results, blending_results_gif])
        
        def feedback_button_click(rating, feedback_form, make_public, input1, input2, extra_images, negative_images, alpha_start, alpha_end, n_steps, blending_results):
            """Handle feedback submission and store to Hugging Face Dataset."""
            if not rating:
                gr.Warning("Please select a rating before submitting feedback.")
                return gr.update(value=None), gr.update(value=""), gr.update(value=True)
            
            # Validate that required images exist
            if input1 is None:
                gr.Warning("Please upload Input 1 image before submitting feedback.")
                return gr.update(value=rating), gr.update(value=feedback_form), gr.update(value=make_public)
            
            if input2 is None:
                gr.Warning("Please upload Input 2 image before submitting feedback.")
                return gr.update(value=rating), gr.update(value=feedback_form), gr.update(value=make_public)
            
            if blending_results is None or len(blending_results) == 0:
                gr.Warning("Please run vibe blending first to generate results before submitting feedback.")
                return gr.update(value=rating), gr.update(value=feedback_form), gr.update(value=make_public)
            
            # Process images to check if they exist
            input1_img, input2_img, extra_images_processed, negative_images_processed = _process_input_images(
                input1, input2, extra_images, negative_images
            )
            
            # Extract images from gallery (gallery returns list of tuples (image_path, caption) or just image paths)
            blending_result_images = []
            for item in blending_results:
                if isinstance(item, tuple):
                    image_path = item[0]
                else:
                    image_path = item
                blending_result_images.append(Image.open(image_path).convert("RGB"))
            
            # Store feedback and images to Hugging Face Dataset (upload list of images)
            success = store_feedback_to_hf_dataset(
                rating=rating,
                feedback_text=feedback_form or "",
                alpha_start=alpha_start,
                alpha_end=alpha_end,
                n_steps=n_steps,
                input1_image=input1_img,
                input2_image=input2_img,
                extra_images=extra_images_processed if extra_images_processed else None,
                negative_images=negative_images_processed if negative_images_processed else None,
                blending_result_images=blending_result_images,  # Upload list of images
                is_public=make_public,  # Pass the public flag
            )
            
            if success:
                gr.Info("Thank you! Your feedback has been submitted successfully.")
                return gr.update(value=None), gr.update(value=""), gr.update(value=True)  # Reset rating, feedback form, and checkbox
            else:
                error_msg = "Feedback could not be stored. "
                if not HF_FEEDBACK_DATASET_REPO:
                    error_msg += "Please configure `HF_FEEDBACK_DATASET_REPO` environment variable (e.g., 'your-username/vibe-blending-feedback')."
                else:
                    error_msg += "Please check the logs for details."
                gr.Warning(error_msg)
                return gr.update(value=rating), gr.update(value=feedback_form), gr.update(value=make_public)  # Keep rating, feedback form, and checkbox
        
        feedback_button.click(
            feedback_button_click,
            inputs=[rating, feedback_form, make_public, input1, input2, extra_images, negative_images, alpha_start, alpha_end, n_steps, blending_results],
            outputs=[rating, feedback_form, make_public]  # Reset rating, feedback form, and checkbox
        )
        
        example_cases = [
            [Image.open("./images/playviolin_hr.png"), Image.open("./images/playguitar_hr.png")],
            [Image.open("./images/input_cat.png"), Image.open("./images/input_bread.png")],
            [Image.open("./images/02140_left.jpg"), Image.open("./images/02140_right.jpg")],
            #[Image.open("./images/02718_l.jpg"), Image.open("./images/02718_r.jpg")],
            [Image.open("./images/03969_l.jpg"), Image.open("./images/03969_r.jpg")],
            [Image.open("./images/04963_l.jpg"), Image.open("./images/04963_r.jpg")],
            #[Image.open("./images/05358_l.jpg"), Image.open("./images/05358_r.jpg")],
            [Image.open("./images/00436_l.jpg"), Image.open("./images/00436_r.jpg")],
            [Image.open("./images/archi/input_A.jpg"), Image.open("./images/archi/input_B.jpg")],
        ]
        gr.Examples(examples=example_cases, label="Example Cases", inputs=[input1, input2], outputs=[blending_results_grid, blending_results])
        
        extra_image_examples = [
            [Image.open("./images/archi/input_A.jpg"), Image.open("./images/archi/input_B.jpg"), [Image.open("./images/archi/extra1.jpg"), Image.open("./images/archi/extra2.jpg"), Image.open("./images/archi/extra3.jpg")]],
        ]
        gr.Examples(examples=extra_image_examples, label="Extra Image Examples", inputs=[input1, input2, extra_images], outputs=[blending_results_grid, blending_results])
        
        negative_image_examples = [
            [Image.open("./images/pink_bear1.jpg"), Image.open("./images/black_bear2.jpg"), [Image.open("./images/pink_bear1.jpg"), Image.open("./images/black_bear1.jpg")]],
        ]
        gr.Examples(examples=negative_image_examples, label="Negative Image Examples", inputs=[input1, input2, negative_images], outputs=[blending_results_grid, blending_results])


# Feedback viewer functions moved to feedback_viewer.py


def create_merged_interface():
    """Create merged interface with both tabs."""
    theme = gr.themes.Base(
        spacing_size='md', 
        text_size='lg', 
        primary_hue='blue', 
        neutral_hue='slate', 
        secondary_hue='pink'
    )
    
    demo = gr.Blocks(theme=theme)
    with demo:
        # Create both tabs
        create_vibe_blending_tab()
        create_feedback_viewer_tab()
    
    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    demo = create_merged_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0" if USE_HUGGINGFACE_ZEROGPU else None,
        show_error=True
    )
