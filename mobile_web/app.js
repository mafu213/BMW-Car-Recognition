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
const debugStatus = document.getElementById("debugStatus");

const video = document.getElementById("video");
const captureCanvas = document.getElementById("captureCanvas");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const cameraHint = document.getElementById("cameraHint");
const openCameraBtn = document.getElementById("openCameraBtn");
const captureBtn = document.getElementById("captureBtn");
const retakeBtn = document.getElementById("retakeBtn");
const cameraFile = document.getElementById("cameraFile");
const uploadPredictBtn = document.getElementById("uploadPredictBtn");
const testModelBtn = document.getElementById("testModelBtn");
const previewImage = document.getElementById("previewImage");
const previewPlaceholder = document.getElementById("previewPlaceholder");

let session = null;
let idxToClass = {};
let stream = null;
let selectedImage = null;

function logStep(message, data) {
  const text = data === undefined ? message : `${message}: ${JSON.stringify(data)}`;
  console.log(`[BMW] ${message}`, data === undefined ? "" : data);
  debugStatus.textContent = text;
}

function setReady(text, ok) {
  modelStatus.textContent = text;
  readyBadge.textContent = ok ? "模型就绪" : "准备中";
  readyBadge.className = ok ? "status ready" : "status";
  logStep(text);
}

function setError(text, error) {
  const detail = error && (error.message || String(error)) ? `：${error.message || String(error)}` : "";
  const fullText = `${text}${detail}`;
  modelStatus.textContent = fullText;
  readyBadge.textContent = "不可用";
  readyBadge.className = "status error";
  predLabel.textContent = fullText;
  confidence.textContent = "-";
  topkBars.innerHTML = "";
  debugStatus.textContent = fullText;
  console.error(`[BMW] ${fullText}`, error || "");
}

function setWorking(text) {
  predLabel.textContent = text;
  confidence.textContent = "-";
  topkBars.innerHTML = "";
  logStep(text);
}

function displayName(classId) {
  return `${classId} - ${CLASS_NAMES[classId] || "BMW Class"}`;
}

function percent(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function softmax(logits) {
  const values = Array.from(logits);
  const maxValue = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - maxValue));
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
  if (!topk.length) {
    throw new Error("Top-4 结果为空");
  }
  const best = topk[0];
  predLabel.textContent = best.label;
  confidence.textContent = percent(best.prob);
  renderTopK(topk);
  logStep("结果已显示", topk);
}

function showPreviewFromCanvas(canvas) {
  previewImage.src = canvas.toDataURL("image/jpeg", 0.92);
  previewImage.style.display = "block";
  previewPlaceholder.hidden = true;
}

function drawSourceToSquareCanvas(source) {
  const canvas = document.createElement("canvas");
  canvas.width = IMG_SIZE;
  canvas.height = IMG_SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    throw new Error("无法创建 2D canvas context");
  }
  ctx.drawImage(source, 0, 0, IMG_SIZE, IMG_SIZE);
  return canvas;
}

function preprocess(source) {
  try {
    const canvas = drawSourceToSquareCanvas(source);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const imageData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data;
    if (!imageData || imageData.length !== IMG_SIZE * IMG_SIZE * 4) {
      throw new Error("canvas image data empty");
    }

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
    logStep("图像预处理完成", { shape: [1, 3, IMG_SIZE, IMG_SIZE], length: input.length });
    console.log("[BMW] 输入 tensor shape", [1, 3, IMG_SIZE, IMG_SIZE]);
    return input;
  } catch (error) {
    throw new Error(`预处理失败：${error.message || error}`);
  }
}

async function predictImage(source) {
  if (!session) {
    throw new Error("模型未加载完成");
  }

  setWorking("开始推理准备...");
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  if (!inputName) {
    throw new Error("ONNX input name not found");
  }
  if (!outputName) {
    throw new Error("ONNX output name not found");
  }
  console.log("[BMW] ONNX inputNames", session.inputNames);
  console.log("[BMW] ONNX outputNames", session.outputNames);

  const input = preprocess(source);
  const tensor = new ort.Tensor("float32", input, [1, 3, IMG_SIZE, IMG_SIZE]);
  logStep("开始推理", { inputName, outputName });
  const results = await session.run({ [inputName]: tensor });
  const output = results[outputName];
  if (!output || !output.data) {
    throw new Error(`推理失败：output tensor not found (${outputName})`);
  }

  const logits = Array.from(output.data);
  console.log("[BMW] logits", logits);
  const probs = softmax(logits);
  console.log("[BMW] softmax 概率", probs);
  const topk = probs
    .map((prob, index) => {
      const classId = idxToClass[String(index)] || String(index);
      return { label: displayName(classId), prob };
    })
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 4);
  console.log("[BMW] Top-4 结果", topk);
  logStep("推理完成", { outputName, logitsLength: logits.length });
  renderResult(topk);
}

async function captureAndPredict() {
  try {
    if (!video.videoWidth || !video.videoHeight) {
      throw new Error(`canvas 截图失败：video size is ${video.videoWidth}x${video.videoHeight}`);
    }
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    const ctx = captureCanvas.getContext("2d");
    if (!ctx) {
      throw new Error("canvas 截图失败：无法创建 2D context");
    }
    ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    showPreviewFromCanvas(captureCanvas);
    logStep("已拍照", { width: captureCanvas.width, height: captureCanvas.height });
    await predictImage(captureCanvas);
  } catch (error) {
    setError("推理失败", error);
  }
}

async function loadModel() {
  try {
    if (location.protocol === "file:") {
      setError("请通过 HTTP/HTTPS 打开，本地 file:// 无法加载模型。");
      return;
    }
    logStep("模型加载中");
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
    console.log("[BMW] ONNX inputNames", session.inputNames);
    console.log("[BMW] ONNX outputNames", session.outputNames);
    setReady("模型加载完成，可拍照识别", true);
  } catch (error) {
    setError("模型加载失败", error);
  }
}

async function openCamera() {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("当前浏览器不支持 getUserMedia");
    }
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    logStep("正在打开摄像头");
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    await new Promise((resolve) => {
      if (video.videoWidth && video.videoHeight) {
        resolve();
      } else {
        video.onloadedmetadata = resolve;
      }
    });
    cameraPlaceholder.hidden = true;
    cameraPlaceholder.style.display = "none";
    cameraHint.hidden = true;
    captureBtn.disabled = false;
    logStep("摄像头打开成功", { width: video.videoWidth, height: video.videoHeight });
  } catch (error) {
    cameraHint.hidden = false;
    captureBtn.disabled = true;
    setError("摄像头打开失败", error);
  }
}

function resetResult() {
  predLabel.textContent = "等待识别";
  confidence.textContent = "-";
  topkBars.innerHTML = "";
  logStep("等待操作");
}

async function loadFileToImage(file) {
  return new Promise((resolve, reject) => {
    if (!file) {
      reject(new Error("未选择图片"));
      return;
    }
    const image = new Image();
    const url = URL.createObjectURL(file);
    image.onload = () => {
      previewImage.src = url;
      previewImage.style.display = "block";
      previewPlaceholder.hidden = true;
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("图片读取失败，请重新拍照。"));
    };
    image.src = url;
  });
}

async function uploadAndPredict() {
  try {
    if (!selectedImage) {
      throw new Error("请先拍照或选择图片");
    }
    logStep("使用上传图片识别");
    await predictImage(selectedImage);
  } catch (error) {
    setError("推理失败", error);
  }
}

function createDemoCanvas() {
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 420;
  const ctx = canvas.getContext("2d");
  const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
  gradient.addColorStop(0, "#f97316");
  gradient.addColorStop(1, "#111827");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 42px Arial";
  ctx.fillText("BMW demo test", 110, 210);
  return canvas;
}

async function testModelInference() {
  try {
    const source = selectedImage || createDemoCanvas();
    if (!selectedImage) {
      showPreviewFromCanvas(source);
    }
    logStep(selectedImage ? "测试模型推理：使用上传图片" : "测试模型推理：使用内置 demo canvas");
    await predictImage(source);
  } catch (error) {
    setError("测试推理失败", error);
  }
}

if (!openCameraBtn || !captureBtn) {
  setError("按钮绑定失败", new Error("openCameraBtn 或 captureBtn 不存在"));
} else {
  openCameraBtn.addEventListener("click", openCamera);
  captureBtn.addEventListener("click", captureAndPredict);
}
retakeBtn.addEventListener("click", resetResult);
uploadPredictBtn.addEventListener("click", uploadAndPredict);
testModelBtn.addEventListener("click", testModelInference);
cameraFile.addEventListener("change", async () => {
  try {
    const file = cameraFile.files && cameraFile.files[0] ? cameraFile.files[0] : null;
    selectedImage = await loadFileToImage(file);
    uploadPredictBtn.disabled = false;
    logStep("已选择照片，可点击上传照片识别", { width: selectedImage.naturalWidth, height: selectedImage.naturalHeight });
  } catch (error) {
    selectedImage = null;
    uploadPredictBtn.disabled = true;
    setError("图片读取失败", error);
  }
});

if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
  cameraHint.hidden = false;
}

loadModel();
