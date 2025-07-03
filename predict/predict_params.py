

import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

# Add parent directory to path to import model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.cnn_model import TuringCNN

# === CONFIG ===
MODEL_PATH = 'results/turing_cnn.pth'
IMAGE_SIZE = 128  # Should match training size

# === DEVICE ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === LOAD MODEL ===
model = TuringCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# === IMAGE TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(),  # Ensure it's 1 channel
    transforms.ToTensor()
])

# === PARAMETER RANGES ===
Du_range = (0.10, 0.25)
Dv_range = (0.05, 0.20)
F_range  = (0.02, 0.09)
k_range  = (0.03, 0.07)

# ✅ FUNCTION TO CALL FROM WEB
def predict_parameters(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy()[0]

    Du = Du_range[0] + prediction[0] * (Du_range[1] - Du_range[0])
    Dv = Dv_range[0] + prediction[1] * (Dv_range[1] - Dv_range[0])
    F  = F_range[0]  + prediction[2] * (F_range[1]  - F_range[0])
    k  = k_range[0]  + prediction[3] * (k_range[1]  - k_range[0])

    return Du, Dv, F, k

# (Optional) Test run when this file is directly run
if __name__ == "__main__":
    test_image_path = 'test_images/tiger.jpg'
    Du, Dv, F, k = predict_parameters(test_image_path)
    print("\n🧠 Predicted Turing Pattern Parameters:")
    print(f"   D_u ≈ {Du:.5f}")
    print(f"   D_v ≈ {Dv:.5f}")
    print(f"   F   ≈ {F:.5f}")
    print(f"   k   ≈ {k:.5f}")
