
import os
import json
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
import sys

path_to_folder_of_test_images = sys.argv[1]
path_to_output_file = sys.argv[2]

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)

processor = AutoProcessor.from_pretrained(model_id)

image_folder = path_to_folder_of_test_images
output_json_file = path_to_output_file

image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

results = {}

total = 0

for image_name in tqdm(image_files):
    image_path = os.path.join(image_folder, image_name)

    raw_image = Image.open(image_path)

    inputs = processor(images=raw_image, text="USER: <image> Provide a one-sentence description of this image. ASSISTANT:", return_tensors='pt').to(0, torch.float16)

    output = model.generate(**inputs, max_new_tokens=50, num_beams=3, do_sample=False)  

    description = processor.decode(output[0][2:], skip_special_tokens=True)
    description = description.split("ASSISTANT: ")[-1]

    no_jpg = image_name.split('.')[0]
    results[no_jpg] = description

    total += 1

print(total)

with open(output_json_file, 'w') as f:
    json.dump(results, f, indent=4)

