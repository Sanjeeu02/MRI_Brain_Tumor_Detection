import os, shutil, random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import gradio as gr

# =====================
# 1️⃣ Dataset Setup
# =====================
base_path = r"C:\Users\DELL 0424\Downloads\archive"
original_yes = os.path.join(base_path, "yes")
original_no = os.path.join(base_path, "no")

train_normal = "dataset/train/Normal"
train_abnormal = "dataset/train/Abnormal"
test_normal = "dataset/test/Normal"
test_abnormal = "dataset/test/Abnormal"

os.makedirs(train_normal, exist_ok=True)
os.makedirs(train_abnormal, exist_ok=True)
os.makedirs(test_normal, exist_ok=True)
os.makedirs(test_abnormal, exist_ok=True)

def split_data(src, train_dst, test_dst, split_ratio=0.8):
    files = os.listdir(src)
    random.shuffle(files)
    split = int(len(files) * split_ratio)
    train_files = files[:split]
    test_files = files[split:]

    for f in train_files:
        shutil.copy(os.path.join(src, f), train_dst)
    for f in test_files:
        shutil.copy(os.path.join(src, f), test_dst)

if not os.listdir(train_normal):  # run only if dataset not split yet
    split_data(original_yes, train_abnormal, test_abnormal)
    split_data(original_no, train_normal, test_normal)
    print("✅ Dataset prepared")

# =====================
# 2️⃣ Model
# =====================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

# =====================
# 3️⃣ Training (Run once)
# =====================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = ImageFolder("dataset/train", transform=transform)
test_data = ImageFolder("dataset/test", transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model_path = r"C:\Users\DELL 0424\Downloads\archive\coding\tumor_model.pth"

if not os.path.exists(model_path):
    print("🚀 Training model...")
    for epoch in range(3):  # keep small for testing
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} done ✅")
    torch.save(model.state_dict(), model_path)
    print("💾 Model saved at", model_path)
else:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("📂 Pre-trained model loaded")

model.eval()

# =====================
# 4️⃣ Prediction
# =====================
def predict(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
    return "🟢 Normal" if pred.item() == 0 else "🔴 Abnormal (Tumor Detected)"

# =====================
# 5️⃣ Gradio UI
# =====================
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="label",
    title="🧠 MRI Tumor Detection AI",
    description="Upload an MRI brain image to detect tumor."
)

interface.launch(server_name="0.0.0.0", server_port=7860,share=True)
