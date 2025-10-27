from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import io
from PIL import Image
import json
import torch
from torchvision import transforms, models

app = FastAPI(title='Skin Tone Predictor')

# load palette if present
PALETTE_PATH = '/content/fashion_palette.json'
try:
    with open(PALETTE_PATH, 'r') as f:
        PALETTE = json.load(f)
except Exception:
    PALETTE = {}

# NOTE: set CKPT_PATH and CLASSES before running server or load dynamically
CKPT_PATH = '/content/data/model/skin_tone_cnn.pt'
CLASSES = None
MODEL = None


def load_model(ckpt_path, classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.efficientnet_b0(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, len(classes))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


@app.on_event('startup')
def startup_event():
    global MODEL, CLASSES
    # try to load classes.json
    try:
        CLASSES = json.load(open('/content/data/classes.json'))
    except Exception:
        CLASSES = None
    if CLASSES and MODEL is None:
        try:
            MODEL = load_model(CKPT_PATH, CLASSES)
        except Exception:
            MODEL = None


def transform_image(img_bytes):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return tf(image).unsqueeze(0)


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    x = transform_image(data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if MODEL is None or CLASSES is None:
        return JSONResponse({'error': 'model or classes not loaded on server'}, status_code=500)
    x = x.to(device)
    with torch.no_grad():
        out = MODEL(x)
        pred = out.argmax(1).item()
    tone = CLASSES[pred]
    return {'tone': tone, 'palette': PALETTE.get(tone, ['black', 'white'])}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
