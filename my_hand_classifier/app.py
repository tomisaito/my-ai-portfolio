
import gradio as gr
import torch
from torchvision import transforms
from model import create_model

# 設定
class_names = ['fleming_left', 'open_palm', 'thumbs_up'] # 順序は確認済みと仮定
model = create_model(num_classes=len(class_names), pretrained=False)
model.load_state_dict(torch.load("best_model.pth", map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    if image is None: return None
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Left Hand Classifier",
    description="Classify: Fleming / Open Palm / Thumbs Up"
)

if __name__ == "__main__":
    demo.launch()
