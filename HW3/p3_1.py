import os
import json
from PIL import Image
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from transformers import AutoProcessor, LlavaForConditionalGeneration
import numpy as np

def create_attention_visualization(image, attention_weights, token, save_path=None):
    """
    Creates a visualization of attention weights for a single token and head.
    """
    # Create figure
    plt.figure(figsize=(5, 4))
    plt.subplots_adjust(top=0.9, bottom=0, left=0, right=1)
    
    # Reshape attention weights to match image patches (24x24)
    attention_map = attention_weights.reshape(24, 24)


    image_cpu = image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
    image_cpu = (image_cpu - image_cpu.min()) / (image_cpu.max() - image_cpu.min() + 1e-8)  # Normalize
    image_cpu = (image_cpu).astype(np.float32)
    # Create attention overlay
    plt.subplot(1, 1, 1)
    plt.imshow(image_cpu)
    
    # Resize attention map to match image size
    h, w = image_cpu.shape[1], image_cpu.shape[0]
    attention_resized = torch.nn.functional.interpolate(
        torch.tensor(attention_map, dtype=torch.float32)[None, None],
        size=(h, w),
        mode='bilinear',
        align_corners=False
    )[0, 0].numpy()
    
    attention_resized = (attention_resized - attention_resized.min()) / \
                        (attention_resized.max() - attention_resized.min() + 1e-8)
    
    # Apply attention overlay with jet colormap
    plt.imshow(attention_resized, alpha=0.5, cmap='jet', vmin=0, vmax=1)
    plt.title(f'{token}', y=0.98, pad=20,  fontsize=20, fontweight='bold', va='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_image_with_attention(model, processor, image_path, output_dir):
    """
    Process single image and generate attention visualizations for each token and head.
    """
    raw_image = Image.open(image_path)
    
    inputs = processor(images=raw_image, text="USER: <image> Provide a one-sentence description of this image. ASSISTANT:", return_tensors='pt').to(0, torch.float16)
    
    # Print input tokens for debugging
    input_tokens = processor.tokenizer.convert_ids_to_tokens(inputs.input_ids[0].cpu().tolist())
    
    image_start = input_tokens.index('<image>') + 1
    image_end = image_start + 576
    
    # Generate caption
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        num_beams=1,
        do_sample=False,
        output_attentions=True,
        return_dict_in_generate=True
    )

    croped_img = inputs['pixel_values']
    
    # Get full output sequence and find tokens
    output_sequence = outputs.sequences[0]
    
    # Get the description text and its tokens
    description = processor.decode(output_sequence, skip_special_tokens=True)
    description = description.split("ASSISTANT: ")[-1].strip()
    print(f"\nDescription: {description}")
    
    # Encode the description to get actual tokens
    description_ids = processor.tokenizer.encode(description, add_special_tokens=False)
    description_tokens = processor.tokenizer.convert_ids_to_tokens(description_ids)
    print("\nDescription tokens:", description_tokens)
    
    # Process each token
    for token_idx, token in enumerate(description_tokens):
        # Calculate position in attention sequence
        attention_idx = token_idx
            
        # Get attention for this token from the last layer
        head_idx = 16
        attention_weights = outputs.attentions[attention_idx][-1][0, head_idx, 0]
        image_attention = attention_weights[image_start:image_end].float().cpu().numpy()
        
        save_path = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(image_path))[0]}_"
            f"token_{token_idx:02d}_{token}_head_{head_idx:02d}.png"
        )
        
        create_attention_visualization(
            croped_img, 
            image_attention, 
            token,
            save_path
        )
    
    return description

def main():
    model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        output_attentions=True
    ).to(0)
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    image_folder = './hw3_data/p3_data/images'
    output_json_file = 'p3_output.json'
    output_vis_dir = './attention_visualizations'
    os.makedirs(output_vis_dir, exist_ok=True)
    
    # 取得所有圖片
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    results = {}
    
    # 使用tqdm來顯示處理進度
    for image_name in tqdm(image_files, desc="Processing images"):
        print(f"\nProcessing {image_name}...")
        image_path = os.path.join(image_folder, image_name)
        no_jpg = image_name.split('.')[0]
        
        description = process_image_with_attention(
            model, 
            processor, 
            image_path,
            output_vis_dir
        )
        results[no_jpg] = description
            
    # 儲存所有結果
    with open(output_json_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()