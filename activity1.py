import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer, pipeline
from diffusers import StableDiffusionPipeline
import gradio as gr
from PIL import Image

print("Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
text_generator = pipeline('text-generation', model=gpt2_model, tokenizer=gpt2_tokenizer, device=0 if device=="cuda" else -1)
sd_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16 if device=="cuda" else torch.float32)
sd_pipe = sd_pipe.to(device)

def caption_image(input_image):
    """Step 1: Generate simple caption from image"""
    inputs = blip_processor(input_image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_length=50)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

def expand_caption(caption):
    """Step 2: Expand caption into descriptive story using GPT-2"""
    prompt = f"Create a detailed, vivid description of this scene: {caption}. The detailed description is:"
    result = text_generator(prompt, max_length=150, num_return_sequences=1, temperature=0.8, do_sample=True, pad_token_id=gpt2_tokenizer.eos_token_id)
    expanded_text = result[0]['generated_text'].replace(prompt, "").strip()
    return expanded_text

def generate_new_image(prompt):
    """Step 3: Generate new image from expanded text"""
    image = sd_pipe(prompt, guidance_scale=7.5, num_inference_steps=30).images[0]
    return image

def full_workflow(input_image):
    simple_caption = caption_image(input_image)
    expanded_description = expand_caption(simple_caption)
    full_prompt = f"{simple_caption}, {expanded_description}"
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(full_prompt, max_length=50, min_length=20, do_sample=False)[0]['summary_text']
    new_image = generate_new_image(full_prompt[:300])

    return simple_caption, expanded_description, summary, new_image
with gr.Blocks(title="AI Image-to-Text-to-Image Project - Part 2") as demo:
    gr.Markdown("# AI-Powered Image-to-Text Conversion Project – Part 2")
    gr.Markdown("Upload an image -> Get Caption -> Expand with GPT-2 -> Generate new image with Stable Diffusion")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Upload Image")
            btn = gr.Button("Run Complete Workflow")
        with gr.Column():
            out_caption = gr.Textbox(label="Step 1: Simple Caption (BLIP)")
            out_expanded = gr.Textbox(label="Step 2: Expanded Descriptive Text (GPT-2)", lines=4)
            out_summary = gr.Textbox(label="Step 3: Summary of Content", lines=2)
            out_image = gr.Image(label="Step 4: New Image from Text (Stable Diffusion)")

    btn.click(fn=full_workflow, inputs=input_img, outputs=[out_caption, out_expanded, out_summary, out_image])

demo.launch()