from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import uvicorn

app = FastAPI(title="Image Classifier API")

def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
    # ... (same as in the second script)

# Load the model and the class labels
model = ResNet9(in_channels=3, num_classes=6)
model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4951, 0.4982, 0.4979], std=[0.2482, 0.2467, 0.2807])
])
class_labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def predict_class(image_path):
    input_image = Image.open(image_path)
    input_image = test_transform(input_image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_image)
        _, predicted_index = torch.max(output, 1)
        predicted_class = class_labels[predicted_index.item()]
    return predicted_class

@app.post("/predict/tf/", status_code=200)
async def predict_tf(image: UploadFile = File(...)):
    try:
        # Save the uploaded image to a temporary file
        with open("temp_image.jpg", "wb") as f:
            f.write(image.file.read())
        predicted_class = predict_class("temp_image.jpg")
        return {"predicted_class": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=404, detail="Image could not be processed")

if __name__ == "__main__":
    uvicorn.run("app.app:app", host="127.0.0.1", port=8000, log_level="debug",
                proxy_headers=True, reload=True)
