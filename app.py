import io
import json
import os
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from train import build_model, get_eval_transform


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
CLASS_TO_IDX_PATH = CHECKPOINT_DIR / "class_to_idx.json"
DESCRIPTION_PATH = CHECKPOINT_DIR / "class_descriptions.json"

app = FastAPI(title="BMW Car Recognition")
app.mount("/static", StaticFiles(directory=str(PROJECT_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(PROJECT_DIR / "templates"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
idx_to_class = {}
class_descriptions = {}
img_size = 224
model_name = "mobilenet_v3_small"
load_error = None


def label_for(class_name):
    description = class_descriptions.get(class_name, "")
    return f"{class_name} - {description}" if description else class_name


def load_model_once():
    """启动时加载模型；失败时保留错误信息给网页显示。"""
    global model, idx_to_class, class_descriptions, img_size, model_name, load_error
    if model is not None or load_error is not None:
        return

    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"缺少模型文件：{MODEL_PATH}")
        if not CLASS_TO_IDX_PATH.exists():
            raise FileNotFoundError(f"缺少类别映射文件：{CLASS_TO_IDX_PATH}")

        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            model_name = checkpoint.get("model_name", model_name)
            img_size = int(checkpoint.get("img_size", img_size))
        else:
            state_dict = checkpoint

        class_to_idx = json.loads(CLASS_TO_IDX_PATH.read_text(encoding="utf-8"))
        idx_to_class = {int(idx): class_name for class_name, idx in class_to_idx.items()}
        if DESCRIPTION_PATH.exists():
            class_descriptions = json.loads(DESCRIPTION_PATH.read_text(encoding="utf-8"))

        loaded_model = build_model(len(idx_to_class), model_name=model_name, use_pretrained=False)
        loaded_model.load_state_dict(state_dict, strict=True)
        loaded_model.to(device)
        loaded_model.eval()
        model = loaded_model
        load_error = None
        print(f"[网页] 模型加载成功：{MODEL_PATH}，设备：{device}，结构：{model_name}")
    except Exception as exc:
        load_error = str(exc)
        print(f"[网页] 模型加载失败：{load_error}")


@app.on_event("startup")
def startup_event():
    load_model_once()


@app.get("/")
def index(request: Request):
    load_model_once()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_ready": model is not None,
            "load_error": load_error,
            "device": str(device),
            "model_name": model_name,
        },
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """接收普通上传、canvas JPEG Blob、iPhone 拍照文件并返回 Top-4 概率。"""
    load_model_once()
    if model is None:
        raise HTTPException(status_code=500, detail=f"模型未加载：{load_error}")

    content_type = file.content_type or ""
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件。")

    try:
        content = await file.read()
        if not content:
            raise ValueError("图片内容为空")
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"图片读取失败：{exc}") from exc

    transform = get_eval_transform(img_size)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    top_k = min(4, len(idx_to_class))
    values, indices = torch.topk(probs, k=top_k)

    topk = []
    top4 = []
    for value, index in zip(values.tolist(), indices.tolist()):
        class_name = idx_to_class[int(index)]
        display_name = label_for(class_name)
        probability = float(value)
        topk.append({"label": display_name, "prob": probability})
        top4.append({
            "class_name": class_name,
            "display_name": display_name,
            "probability": probability,
        })

    best = top4[0]
    return JSONResponse({
        # 新网页和验收文档使用的字段
        "pred_label": best["display_name"],
        "confidence": best["probability"],
        "topk": topk,
        # 兼容旧网页或旧脚本的字段
        "predicted_class": best["class_name"],
        "display_name": best["display_name"],
        "top4": top4,
        "model_name": model_name,
        "device": str(device),
    })


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    uvicorn.run(app, host="0.0.0.0", port=8000)
