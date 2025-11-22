import gradio as gr
from PIL import Image
import numpy as np
import os

# Assume these are your model and utility imports
# Please adjust the import paths according to your actual setup
# from your_inference_module import inference_controlnet  # Replace with your actual inference module

# Global variable to store uploaded images
uploaded_images = []

def process_images(images, prompt, controlnet_conditioning_scale, msadapter_scale):
    """
    Process uploaded images and generate results
    
    Args:
        images: List of uploaded images
        prompt: Text prompt
        controlnet_conditioning_scale: ControlNet conditioning strength
        msadapter_scale: MSAdapter strength
        
    Returns:
        Path to the generated image
    """
    global uploaded_images
    
    if not images:
        gr.Warning("Please upload at least one image")
        return None
    
    # Process uploaded images
    processed_images = []
    for img in images:
        # Convert to PIL Image and resize
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = img.convert("RGB").resize((512, 512))
        processed_images.append(img)
    
    uploaded_images = processed_images
    
    # Prepare parameters
    result_path = './res/action_result/three_objects'
    os.makedirs(result_path, exist_ok=True)  # Create directory if it doesn't exist
    
    save_name = f'controlnet_{controlnet_conditioning_scale}_msadapter_{msadapter_scale}_result.jpg'
    output_path = os.path.join(result_path, save_name)
    
    # Load control image (using a fixed openpose image, you can modify this to allow user upload)
    try:
        control_image = Image.open("./res/openpose/openpose_2.png").resize((1024, 1024))
    except Exception as e:
        gr.Error(f"Error loading control image: {str(e)}")
        return None
    
    # Prepare boxes (using default boxes, modify as needed)
    boxes = [[[0., 0., 0., 0.] for _ in range(len(processed_images))]]
    
    # Assume phrases are extracted from the prompt, simple processing here
    phrases = prompt.split(',') if ',' in prompt else [prompt]
    
    # Call inference function
    try:
        # Note: Adjust the parameters according to your actual ms_sd_generate_image function
        # This is a mock implementation - replace with your actual function call
        print(f"Generating image with prompt: {prompt}")
        print(f"ControlNet scale: {controlnet_conditioning_scale}, MSAdapter scale: {msadapter_scale}")
        
        # Mock implementation - remove this and use your actual function
        # generated_image = inference_controlnet.ms_sd_generate_image(
        #     processed_images,
        #     prompt,
        #     phrases,
        #     controlnet_conditioning_scale,
        #     msadapter_scale,
        #     control_image,
        #     boxes=boxes,
        #     result_path=result_path,
        #     save_name=save_name
        # )
        
        # For demonstration, save a placeholder image
        placeholder = Image.new('RGB', (512, 512), color='red')
        placeholder.save(output_path)
        return output_path
        
    except Exception as e:
        gr.Error(f"Error generating image: {str(e)}")
        return None

def create_demo():
    """Create Gradio demo interface"""
    with gr.Blocks(title="ControlNet Image Generator") as demo:
        gr.Markdown(
            """
            # ControlNet Image Generator
            Upload images and enter text prompts to generate images using ControlNet.
            Supports multiple image inputs and parameter adjustments.
            """
        )
        
        with gr.Row():
            # Left side: Input area
            with gr.Column(scale=1):
                image_upload = gr.Image(
                    type="pil",
                    label="Upload Images",
                    sources=["upload"],
                    elem_id="image_upload",
                    height=200
                )
                # Enable multiple image uploads
                image_upload.multiple = True
                
                prompt_input = gr.Textbox(
                    label="Text Prompt",
                    placeholder="Enter text describing the image content...",
                    lines=3,
                    value="A Korean woman wearing a blue button-down shirt and a white skirt, carrying a sleek black handbag"
                )
                
                with gr.Accordion("Advanced Parameters", open=False):
                    controlnet_scale = gr.Slider(
                        label="ControlNet Conditioning Scale",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.9,
                        step=0.1,
                        info="Controls the influence strength of ControlNet on the generated result"
                    )
                    
                    msadapter_scale = gr.Slider(
                        label="MSAdapter Scale",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.9,
                        step=0.1,
                        info="Controls the strength of MSAdapter"
                    )
                
                submit_btn = gr.Button("Generate Image", variant="primary", size="lg")
            
            # Right side: Output area
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Generated Image",
                    type="filepath",
                    height=400
                )
                
                output_gallery = gr.Gallery(
                    label="Uploaded Images",
                    show_label=True,
                    elem_id="input_gallery",
                    columns=3,
                    rows=1,
                    height=150
                )
        
        # Example prompts
        gr.Examples(
            examples=[
                [
                    ["examples/example1.jpg"],  # Path to example image
                    "A woman wearing a red dress standing in a park"
                ],
                [
                    ["examples/example2.jpg"],
                    "A man sitting at a desk working on a laptop"
                ]
            ],
            inputs=[image_upload, prompt_input],
            outputs=output_image,
            fn=process_images,
            cache_examples=False
        )
        
        # Event bindings
        image_upload.change(
            lambda x: x if x else [],
            inputs=image_upload,
            outputs=output_gallery
        )
        
        submit_btn.click(
            fn=process_images,
            inputs=[image_upload, prompt_input, controlnet_scale, msadapter_scale],
            outputs=output_image
        )
    
    return demo

def create_simple_demo():
    """Create a simpler version of the interface"""
    with gr.Blocks(title="Simple ControlNet Generator") as demo:
        gr.Markdown("# Simple ControlNet Generator")
        
        with gr.Row():
            with gr.Column():
                images = gr.Image(type="pil", multiple=True, label="Upload Images")
                prompt = gr.Textbox(label="Prompt", lines=2)
                btn = gr.Button("Generate")
            
            with gr.Column():
                output = gr.Image(label="Generated Result")
        
        btn.click(process_images, [images, prompt, gr.State(0.9), gr.State(0.9)], output)
    
    return demo

if __name__ == "__main__":
    # Create and launch the demo
    demo = create_demo()
    
    # Launch the server
    demo.launch(
        server_name="0.0.0.0",  # Allow LAN access
        server_port=7860,       # Port number
        share=True,            # Create public link
        inbrowser=True         # Open browser automatically
    )