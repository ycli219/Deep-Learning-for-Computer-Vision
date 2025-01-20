import os
import clip
import torch
from PIL import Image
import json

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Load the id-to-label mapping
with open('hw2_data/clip_zeroshot/id2label.json', 'r') as f:
    id2label = json.load(f)

# Prepare the text inputs
labels = [f"A photo of {label}." for label in id2label.values()]
text_inputs = torch.cat([clip.tokenize(label) for label in labels]).to(device)

# Define the validation data path
val_path = 'hw2_data/clip_zeroshot/val'

# Initialize counters
correct = 0
total = 0
successful_cases = []
failed_cases = []

# Iterate over the validation dataset
for img_name in os.listdir(val_path):
    img_path = os.path.join(val_path, img_name)
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    class_id = img_name.split('_')[0]

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs, indices = similarity[0].topk(1)

    predicted_id = indices[0].item()
    total += 1
    if str(predicted_id) == class_id:
        correct += 1
        successful_cases.append((img_name, id2label[class_id], id2label[str(predicted_id)]))
    else:
        failed_cases.append((img_name, id2label[class_id], id2label[str(predicted_id)]))

# Calculate accuracy
accuracy = correct / total

# Report results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nSuccessful Cases:")
for case in successful_cases[:5]:
    print(f"Image: {case[0]}, True Label: {case[1]}, Predicted Label: {case[2]}")
print("\nFailed Cases:")
for case in failed_cases[:5]:
    print(f"Image: {case[0]}, True Label: {case[1]}, Predicted Label: {case[2]}")
