import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import requests

# Load ImageNet labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(LABELS_URL).text.strip().split("\n")

# Load model (change this to your model path)
model = models.resnet50(weights=None)

# If using your converted FP16 model:
# state = torch.load("model_cpu.pt", map_location="cpu")
# def to_fp32(obj):
#     if isinstance(obj, torch.Tensor) and obj.dtype == torch.float16:
#         return obj.float()
#     if isinstance(obj, dict):
#         return {k: to_fp32(v) for k, v in obj.items()}
#     return obj
# model.load_state_dict(to_fp32(state))

# For demo, using pretrained weights
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image):
    if image is None:
        return {}
    
    img = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)[0]
    
    top5_probs, top5_indices = torch.topk(probs, 5)
    
    return {labels[idx]: float(prob) for prob, idx in zip(top5_probs, top5_indices)}

# Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="🖼️ ImageNet 1K Classifier",
    description="Upload an image to classify it into one of 1000 ImageNet categories.",
    # examples=["example.jpg"],  # Add local image if needed
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()