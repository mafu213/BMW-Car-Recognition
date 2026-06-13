import json
import socket
from pathlib import Path

import requests


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "checkpoints" / "best_model.pth"
CLASS_MAP_PATH = PROJECT_DIR / "checkpoints" / "class_to_idx.json"
TEST_IMAGE_ROOT = Path(r"D:\BMW\BMW\test")
BASE_URL = "http://127.0.0.1:8000"


def get_lan_ip():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except OSError:
        return "请运行 ipconfig 查询电脑局域网 IP"


def find_test_image():
    if not TEST_IMAGE_ROOT.exists():
        return None
    for path in TEST_IMAGE_ROOT.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return path
    return None


def main():
    print("========== BMW 网页验收检查 ==========")
    print(f"模型文件：{'存在' if MODEL_PATH.exists() else '缺失'} -> {MODEL_PATH}")
    print(f"类别映射：{'存在' if CLASS_MAP_PATH.exists() else '缺失'} -> {CLASS_MAP_PATH}")
    print(f"电脑访问：http://127.0.0.1:8000")
    print(f"iPhone 访问：http://{get_lan_ip()}:8000")

    try:
        response = requests.get(BASE_URL, timeout=8)
        print(f"首页状态：{response.status_code}")
    except Exception as exc:
        print(f"首页检查失败：{exc}")
        print("请先运行：python app.py")
        return

    image_path = find_test_image()
    if image_path is None:
        print("未找到测试图片，跳过 /predict 上传检查。")
        return

    try:
        with image_path.open("rb") as f:
            files = {"file": (image_path.name, f, "image/jpeg")}
            response = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
        print(f"/predict 状态：{response.status_code}")
        data = response.json()
        print("返回示例：")
        print(json.dumps({
            "pred_label": data.get("pred_label"),
            "confidence": data.get("confidence"),
            "topk": data.get("topk", [])[:4],
        }, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"/predict 检查失败：{exc}")

    print("\n手动检查：")
    print("1. iPhone Safari 打开电脑局域网地址。")
    print("2. 优先测试“拍照上传备用区”。")
    print("3. 如果实时摄像头不可用，确认页面出现降级提示。")
    print("4. 确认预测类别、置信度和 Top-4 概率条形图正常显示。")


if __name__ == "__main__":
    main()
