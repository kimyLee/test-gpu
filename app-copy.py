import torch
import requests
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from lavis.common.gradcam import getAttMap
from lavis.models import load_model_and_preprocess

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr
import torch, gc
from gpuinfo import GPUInfo
import psutil
import time

def prepare_data(image, question):
    gc.collect()
    torch.cuda.empty_cache()
    image = vis_processors["eval"](image).unsqueeze(0).to(device)
    question = txt_processors["eval"](question)
    samples = {"image": image, "text_input": [question]}
    return samples

def running_inf(time_start):
    time_end = time.time()
    time_diff = time_end - time_start
    memory = psutil.virtual_memory()
    gpu_utilization, gpu_memory = GPUInfo.gpu_usage()
    gpu_utilization = gpu_utilization[0] if len(gpu_utilization) > 0 else 0
    gpu_memory = gpu_memory[0] if len(gpu_memory) > 0 else 0
    system_info = f"""
    *Memory: {memory.total / (1024 * 1024 * 1024):.2f}GB, used: {memory.percent}%, available: {memory.available / (1024 * 1024 * 1024):.2f}GB.* 
    *Processing time: {time_diff:.5} seconds.*
    *GPU Utilization: {gpu_utilization}%, GPU Memory: {gpu_memory}MiB.*
    """
    return system_info
    
def gradcam_attention(image, question):
    dst_w = 720
    samples = prepare_data(image, question)
    samples = model.forward_itm(samples=samples)
    
    w, h = image.size
    scaling_factor = dst_w / w

    resized_img = image.resize((int(w * scaling_factor), int(h * scaling_factor)))
    norm_img = np.float32(resized_img) / 255
    gradcam = samples['gradcams'].reshape(24,24)

    avg_gradcam = getAttMap(norm_img, gradcam, blur=True)
    return (avg_gradcam * 255).astype(np.uint8)

def generate_cap(image, question, cap_number):
    time_start = time.time()
    samples = prepare_data(image, question)
    samples = model.forward_itm(samples=samples)
    samples = model.forward_cap(samples=samples, num_captions=cap_number, num_patches=5)
    return pd.DataFrame({'Caption': samples['captions'][0][:cap_number]}), running_inf(time_start)

def postprocess(text):
    for i, ans in enumerate(text):
        for j, w in enumerate(ans):
            if w == '.' or w == '\n':
                ans = ans[:j].lower()
                break
    return ans

def generate_answer(image, question):
    time_start = time.time()
    samples = prepare_data(image, question)
    samples = model.forward_itm(samples=samples)
    samples = model.forward_cap(samples=samples, num_captions=5, num_patches=20)
    samples = model.forward_qa_generation(samples)
    Img2Prompt = model.prompts_construction(samples)
    Img2Prompt_input = tokenizer(Img2Prompt, padding='longest', truncation=True, return_tensors="pt").to(device)

    outputs = llm_model.generate(input_ids=Img2Prompt_input.input_ids,
                            attention_mask=Img2Prompt_input.attention_mask,
                            max_length=20+len(Img2Prompt_input.input_ids[0]),
                            return_dict_in_generate=True,
                            output_scores=True
                            )
    pred_answer = tokenizer.batch_decode(outputs.sequences[:, len(Img2Prompt_input.input_ids[0]):])
    pred_answer = postprocess(pred_answer)
    print(pred_answer, type(pred_answer))
    return pred_answer, running_inf(time_start)
    
# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def load_model(model_selection):
    model = AutoModelForCausalLM.from_pretrained(model_selection)
    tokenizer = AutoTokenizer.from_pretrained(model_selection, use_fast=False)
    return model,tokenizer

# Choose LLM to use
# weights for OPT-350M/OPT-6.7B/OPT-13B/OPT-30B/OPT-66B will download automatically
print("Loading Large Language Model (LLM)...")
llm_model, tokenizer = load_model('facebook/opt-350m')  # ~700MB (FP16)
llm_model.to(device)
model, vis_processors, txt_processors = load_model_and_preprocess(name="img2prompt_vqa", model_type="base", is_eval=True, device=device)


# ---- Gradio Layout -----
title = "From Images to Textual Prompts: Zero-shot VQA with Frozen Large Language Models"
df_init = pd.DataFrame(columns=['Caption'])
raw_image = gr.Image(label='Input image', type="pil")
question = gr.Textbox(label="Input question", lines=1, interactive=True)
text_output = gr.Textbox(label="Output Answer")
demo = gr.Blocks(title=title)
demo.encrypt = False
cap_df = gr.DataFrame(value=df_init, label="Caption dataframe", row_count=(0, "dynamic"), max_rows = 20, wrap=True, overflow_row_behaviour='paginate')
memory = psutil.virtual_memory()
system_info = gr.Markdown(f"*Memory: {memory.total / (1024 * 1024 * 1024):.2f}GB, used: {memory.percent}%, available: {memory.available / (1024 * 1024 * 1024):.2f}GB*")

with demo:
    with gr.Row():
        gr.Markdown('''
            <div>
            <h1 style='text-align: center'>From Images to Textual Prompts: Zero-shot VQA with Frozen Large Language Models</h1>
            </div>
            ''')  
    with gr.Row():
        gr.Markdown('''
            ### How to use this space
            ##### 1. Upload your image and fill your question
            ##### 2. Creating caption from your image
            ##### 3. Answering your question based on uploaded image
        ''')
    with gr.Row():
        with gr.Column():  
            raw_image.render()
        with gr.Column():
            question.render()
            number_cap = gr.Number(precision=0, value=5, label="Selected number of caption you want to generate", interactive=True)
    with gr.Row():
      with gr.Column():
            cap_btn = gr.Button("Generate caption")
            cap_btn.click(generate_cap, [raw_image, question, number_cap], [cap_df, system_info])
      with gr.Column():
            anws_btn = gr.Button("Answer")
            anws_btn.click(generate_answer, [raw_image, question], outputs=[text_output, system_info])
    with gr.Row():  
      with gr.Column():  
      #     gradcam_btn = gr.Button("Generate Gradcam")
      #     gradcam_btn.click(gradcam_attention, [raw_image, question], outputs=[avg_gradcam])
            cap_df.render()
      with gr.Column():  
            text_output.render()
            system_info.render()
    with gr.Row():
        examples = gr.Examples(
            examples=
                [["image1.jpg", "What type of bird is this?"],
                 ["image2.jpg", "What type of bike is on the ground?"],
                 ["image3.jpg", "What is the person in the photo wearing?"]],
            label="Examples", 
            inputs=[raw_image, question]
        )

demo.launch(debug=True)