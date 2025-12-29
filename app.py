"""
Stable Diffusion Application
A simple web interface for generating images using Stable Diffusion
"""

import os
import torch
from diffusers import StableDiffusionPipeline
import gradio as gr
from PIL import Image

class StableDiffusionApp:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        """
        Initialize the Stable Diffusion application
        
        Args:
            model_id (str): Hugging Face model ID to use
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        print(f"Using device: {self.device}")
        
    def load_model(self):
        """Load the Stable Diffusion model"""
        print(f"Loading model: {self.model_id}")
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.pipe = self.pipe.to(self.device)
            
            # Enable memory optimizations
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
            
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate_image(self, prompt, negative_prompt="", num_inference_steps=50, 
                      guidance_scale=7.5, width=512, height=512, seed=None):
        """
        Generate an image from a text prompt
        
        Args:
            prompt (str): Text description of the desired image
            negative_prompt (str): Text description of what to avoid
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): How closely to follow the prompt
            width (int): Image width
            height (int): Image height
            seed (int): Random seed for reproducibility
            
        Returns:
            PIL.Image: Generated image
        """
        if self.pipe is None:
            self.load_model()
        
        try:
            # Set seed for reproducibility
            generator = None
            if seed is not None and seed >= 0:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Generate image with automatic mixed precision for CUDA
            if self.device == "cuda":
                with torch.autocast(self.device):
                    result = self.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt if negative_prompt else None,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        generator=generator
                    )
            else:
                # CPU inference without autocast
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator
                )
            
            image = result.images[0]
            return image
            
        except Exception as e:
            print(f"Error generating image: {e}")
            raise

def create_interface(app):
    """Create Gradio interface for the application"""
    
    def generate_wrapper(prompt, negative_prompt, steps, guidance, width, height, seed):
        """Wrapper function for Gradio interface"""
        if not prompt.strip():
            return None
        
        # Convert seed to int or None
        seed_value = None
        if seed >= 0:
            seed_value = int(seed)
        
        image = app.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=int(width),
            height=int(height),
            seed=seed_value
        )
        return image
    
    # Create Gradio interface
    interface = gr.Interface(
        fn=generate_wrapper,
        inputs=[
            gr.Textbox(
                label="Prompt",
                placeholder="Enter your image description here...",
                lines=3
            ),
            gr.Textbox(
                label="Negative Prompt (Optional)",
                placeholder="What to avoid in the image...",
                lines=2,
                value=""
            ),
            gr.Slider(
                minimum=1,
                maximum=100,
                value=50,
                step=1,
                label="Number of Steps"
            ),
            gr.Slider(
                minimum=1,
                maximum=20,
                value=7.5,
                step=0.5,
                label="Guidance Scale"
            ),
            gr.Slider(
                minimum=128,
                maximum=1024,
                value=512,
                step=64,
                label="Width"
            ),
            gr.Slider(
                minimum=128,
                maximum=1024,
                value=512,
                step=64,
                label="Height"
            ),
            gr.Number(
                label="Seed (-1 for random)",
                value=-1
            )
        ],
        outputs=gr.Image(label="Generated Image", type="pil"),
        title="Stable Diffusion Image Generator",
        description="Generate images from text descriptions using Stable Diffusion",
        examples=[
            ["a beautiful sunset over mountains", "", 50, 7.5, 512, 512, -1],
            ["a cute cat playing with a ball of yarn", "", 50, 7.5, 512, 512, -1],
            ["a futuristic city at night with neon lights", "", 50, 7.5, 512, 512, -1],
        ]
    )
    
    return interface

def main():
    """Main function to run the application"""
    print("Initializing Stable Diffusion Application...")
    
    # Create application instance
    app = StableDiffusionApp()
    
    # Load model at startup
    app.load_model()
    
    # Create and launch interface
    interface = create_interface(app)
    interface.launch(share=False, server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
