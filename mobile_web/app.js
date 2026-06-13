const MODEL_PATH = "model/bmw_model.onnx";
const IDX_TO_CLASS_PATH = "model/idx_to_class.json";
const CLASS_NAMES = {
  "28": "BMW 1 Series Coupe 2012",
  "29": "BMW 3 Series Sedan 2012",
  "32": "BMW X5 SUV 2007",
  "37": "BMW X3 SUV 2012",
};
const IMG_SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

const modelStatus = document.getElementById("modelStatus");
const readyBadge = document.getElementById("readyBadge");
const predLabel = document.getElementById("predLabel");
const confidence = document.getElementById("confidence");
const topkBars = document.getElementById("topkBars");

const video = document.getElementById("video");
const captureCanvas = document.getElementById("captureCanvas");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const cameraHint = document.getElementById("cameraHint");
const openCameraBtn = document.getElementById("openCameraBtn");
const captureBtn = document.getElementById("captureBtn");
const retakeBtn = document.getElementById("retakeBtn");
const cameraFile = document.getElementById("cameraFile");
const previewImage = document.getElementById("previewImage");
const previewPlaceholder = document.getElementById("previewPlaceholder");

let session = null;
let idxToClass = {};
let stream = null;

function setReady(text, ok) {
  modelStatus.textContent = text;
  readyBadge.textContent = ok ? "模型就绪" : "准备中";
  readyBadge.className = ok ? "status ready" : "status";
}

function setError(text) {
  modelStatus.textContent = text;
  readyBadge.textContent = "不可用";
  readyBadge.className = "status error";
}

function displayName(classId) {
  return `${classId} - ${CLASS_NAMES[classId] || "BMW Class"}`;
}

function percent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function softmax(logits) {
  const maxValue = Math.max(...logits);
  const exps = Array.from(logits, (v) => Math.exp(v - maxValue));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function renderTopK(items) {
  topkBars.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "bar-row";

    const head = document.createElement("div");
    head.className = "bar-head";

    const label = document.createElement("span");
    label.className = "bar-label";
    label.textContent = item.label;

    const prob = document.createElement("span");
    prob.className = "bar-prob";
    prob.textContent = percent(item.prob);

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = percent(item.prob);

    head.append(label, prob);
    track.append(fill);
    row.append(head, track);
    topkBars.append(row);
  });
}

function renderResult(topk) {
  const best = topk[0];
  predLabel.textContent = best.label;
  confidence.textContent = percent(best.prob);
  renderTopK(topk);
}

function showPreviewFromCanvas(canvas) {
  previewImage.src = canvas.toDataURL("image/jpeg", 0.92);
  previewImage.style.display = "block";
  previewPlaceholder.style.display = "none";
}

function drawSourceToSquareCanvas(source) {
  const canvas = document.createElement("canvas");
  canvas.width = IMG_SIZE;
  canvas.height = IMG_SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(source, 0, 0, IMG_SIZE, IMG_SIZE);
  return canvas;
}

function preprocess(source) {
  const canvas = drawSourceToSquareCanvas(source);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const imageData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data;
  const input = new Float32Array(1 * 3 * IMG_SIZE * IMG_SIZE);
  const planeSize = IMG_SIZE * IMG_SIZE;

  for (let y = 0; y < IMG_SIZE; y += 1) {
    for (let x = 0; x < IMG_SIZE; x += 1) {
      const pixelIndex = (y * IMG_SIZE + x) * 4;
      const outIndex = y * IMG_SIZE + x;
      const r = imageData[pixelIndex] / 255;
      const g = imageData[pixelIndex + 1] / 255;
      const b = imageData[pixelIndex + 2] / 255;
      input[outIndex] = (r - MEAN[0]) / STD[0];
      input[planeSize + outIndex] = (g - MEAN[1]) / STD[1];
      input[2 * planeSize + outIndex] = (b - MEAN[2]) / STD[2];
    }
  }
  return input;
}

async function runInference(source) {
  if (!session) {
    alert("模型仍在加载，请稍等。");
    return;
  }
  predLabel.textContent = "正在识别...";
  confidence.textContent = "-";
  topkBars.innerHTML = "";

  const input = preprocess(source);
  const tensor = new ort.Tensor("float32", input, [1, 3, IMG_SIZE, IMG_SIZE]);
  const output = await session.run({ input: tensor });
  const logits = output.logits.data;
  const probs = softmax(logits);
  const topk = probs
    .map((prob, index) => {
      const classId = idxToClass[String(index)] || String(index);
      return { label: displayName(classId), prob };
    })
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 4);
  renderResult(topk);
}

async function loadModel() {
  try {
    if (location.protocol === "file:") {
      setError("请通过 HTTP/HTTPS 打开，本地 file:// 无法加载模型。");
      return;
    }
    ort.env.wasm.wasmPaths = new URL("vendor/", window.location.href).href;
    ort.env.wasm.numThreads = 1;
    const timeout = new Promise((_, reject) => {
      setTimeout(() => reject(new Error("模型加载超过 45 秒，请刷新页面或切换网络后重试。")), 45000);
    });
    const classResponse = await Promise.race([fetch(IDX_TO_CLASS_PATH), timeout]);
    idxToClass = await classResponse.json();
    session = await Promise.race([ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    }), timeout]);
    setReady("模型已加载，可拍照识别", true);
  } catch (error) {
    console.error(error);
    setError(`模型加载失败：${error.message || error}`);
  }
}

async function openCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    cameraHint.hidden = false;
    return;
  }
  try {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false,
    });
    video.srcObject = stream;
    cameraPlaceholder.hidden = true;
    cameraHint.hidden = true;
    captureBtn.disabled = false;
  } catch (error) {
    console.warn(error);
    cameraHint.hidden = false;
    captureBtn.disabled = true;
  }
}

async function captureAndPredict() {
  if (!video.videoWidth || !video.videoHeight) {
    cameraHint.hidden = false;
    return;
  }
  captureCanvas.width = video.videoWidth;
  captureCanvas.height = video.videoHeight;
  const ctx = captureCanvas.getContext("2d");
  ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  showPreviewFromCanvas(captureCanvas);
  await runInference(captureCanvas);
}

function resetResult() {
  predLabel.textContent = "等待识别";
  confidence.textContent = "-";
  topkBars.innerHTML = "";
}

async function loadFileAndPredict(file) {
  if (!file) {
    return;
  }
  const image = new Image();
  const url = URL.createObjectURL(file);
  image.onload = async () => {
    previewImage.src = url;
    previewImage.style.display = "block";
    previewPlaceholder.style.display = "none";
    await runInference(image);
  };
  image.onerror = () => {
    URL.revokeObjectURL(url);
    alert("图片读取失败，请重新拍照。");
  };
  image.src = url;
}

openCameraBtn.addEventListener("click", openCamera);
captureBtn.addEventListener("click", captureAndPredict);
retakeBtn.addEventListener("click", resetResult);
cameraFile.addEventListener("change", () => {
  const file = cameraFile.files && cameraFile.files[0] ? cameraFile.files[0] : null;
  loadFileAndPredict(file);
});

if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
  cameraHint.hidden = false;
}

loadModel();
