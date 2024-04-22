import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

NUM_CANDIDATES = 5
NUM_CLASSES = 5

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load('Weights/Learned_Model.pth'))
model.eval()

class_names = [
    "Food",
    "Friend",
    "ID_Photo",
    "Scenery",
    "Study"
]

def predict(image_path):
    image = Image.open(image_path)
    tensor_image = transform(image).unsqueeze(0)
    outputs = model(tensor_image)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    top_prob, top_classes = torch.topk(probabilities, NUM_CANDIDATES)
    top_prob_percent = [round(prob.item() * 100, 2) for prob in top_prob[0]]
    predictions = [(class_names[class_idx], prob) for class_idx, prob in zip(top_classes[0], top_prob_percent)]
    results = {}
    for class_name, prob in predictions:
        results[class_name] = prob
    return results

if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"  # テストする画像のパスを指定
    predictions = predict(image_path)
    for class_name, prob in predictions.items():
        print(f"{class_name}: {prob}%")
