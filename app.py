import os
import gradio as gr
from huggingface_hub import InferenceClient

# 1. TOKEN HANDLING (Security Best Practice)
# If deploying to Hugging Face Spaces, add via Settings -> Secrets (Recommended)
# For local testing, set HF_TOKEN env var OR temporarily paste token below
HF_TOKEN = os.environ.get("HF_TOKEN", "your_huggingface_token_here")

# Initialize the client with the specific model and token
# SDXL 1.0 is known for realistic, detailed images
client = InferenceClient(
    "stabilityai/stable-diffusion-xl-base-1.0",
    token=HF_TOKEN if HF_TOKEN != "your_huggingface_token_here" else None
)

def generate_image(prompt):
    """Sends the text prompt to the model and returns the generated PIL image."""
    if not client.token:
        raise gr.Error("API Token missing. Please add your Hugging Face token securely.")
        
    try:
        # Standard serverless inference for image generation
        image = client.text_to_image(prompt)
        return image
    except Exception as e:
        # Provide user-friendly error messages
        raise gr.Error(f"Image generation error: {str(e)}")

# Defined example prompts showcasing SDXL's capabilities
example_prompts = [
    ["Hyperrealistic photo of a futuristic cyberpunk cityscape at night, flying vehicles, neon signs, detailed architecture, cinematic lighting, 8k resolution"],
    ["Intricate ancient stone temple ruins deep within a lush tropical jungle, misty morning, dappled sunlight filtering through trees, overgrown vines, photorealistic"],
    ["A sleek, modern electric sports car driving on a scenic coastal highway in sunny daylight, crystal clear ocean, sharp focus, dynamic composition"],
    ["Expressive oil painting of turbulent ocean waves crashing onto a rocky shore during a dramatic storm, visible brushstrokes, dynamic perspective"],
    ["A majestic dragon perched on a snowy mountain peak overlooking a vast medieval kingdom, highly detailed scales, epic shot, cinematic composition"],
    ["Detailed watercolor illustration of a charming European village street, blooming flowers, quaint stone houses, cobbled path, sunny day"]
]

# Create a more structured and modern Gradio interface
theme = gr.themes.Soft()  # Use a softer, more modern theme

with gr.Blocks(theme=theme, title="Modern AI Image Generator") as app:
    gr.Markdown("# 🎨 Modern Text-to-Image Studio")
    gr.Markdown("Instantly transform your descriptions into high-quality images using Stable Diffusion XL 1.0.")
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Image Prompt", 
                placeholder="Describe your creative vision in detail... (e.g., A futuristic park with waterfalls)",
                lines=5,
                elem_id="prompt-input"
            )
            gr.Markdown("### 💡 Try these example prompts:")
            # Use gr.Examples component for seamless integration
            gr.Examples(examples=example_prompts, inputs=prompt_input, label=None)

            generate_btn = gr.Button("Generate Stunning Image ✨", variant="primary", scale=1)
            
        with gr.Column(scale=2):
            image_output = gr.Image(label="Generated Artistic Vision", type="pil", interactive=False)

    # Connect the button click event to the generation function
    generate_btn.click(
        fn=generate_image, 
        inputs=prompt_input, 
        outputs=image_output
    )

if __name__ == "__main__":
    app.launch()
