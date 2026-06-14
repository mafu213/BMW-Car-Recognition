const MODEL_DISPLAY_NAME = "EfficientNet-B0 ONNX";
const MODEL_URL = new URL("./model/bmw_model.onnx", window.location.href).href;
const IDX_TO_CLASS_URL = new URL("./model/idx_to_class.json", window.location.href).href;
const VENDOR_URL = new URL("./vendor/", window.location.href).href;
const IMG_SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const CLASS_NAMES = {
  "28": "BMW 1 Series Coupe 2012",
  "29": "BMW 3 Series Sedan 2012",
  "32": "BMW X5 SUV 2007",
  "37": "BMW X3 SUV 2012",
};

const modelStatus = document.getElementById("modelStatus");
const readyBadge = document.getElementById("readyBadge");
const modelDetails = document.getElementById("modelDetails");
const debugStatus = document.getElementById("debugStatus");
const video = document.getElementById("video");
const captureCanvas = document.getElementById("captureCanvas");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const openCameraBtn = document.getElementById("openCameraBtn");
const captureBtn = document.getElementById("captureBtn");
const resetBtn = document.getElementById("resetBtn");
const localFile = document.getElementById("localFile");
const localPredictBtn = document.getElementById("localPredictBtn");
const previewImage = document.getElementById("previewImage");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const predLabel = document.getElementById("predLabel");
const confidence = document.getElementById("confidence");
const topkBars = document.getElementById("topkBars");

let session = null;
let idxToClass = {};
let stream = null;
let selectedImage = null;
let modelSizeBytes = 0;

function safeJson(data) {
  try {
    return JSON.stringify(data, null, 2);
  } catch (error) {
    return String(data);
  }
}

function setDebug(message, data) {
  const detail = data === undefined ? "" : `\n${safeJson(data)}`;
  debugStatus.textContent = `${new Date().toLocaleTimeString()} ${message}${detail}`;
  console.log(`[BMW Desktop] ${message}`, data === undefined ? "" : data);
}

function formatMb(bytes) {
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

function renderModelDetails(extra = {}) {
  const lines = [
    `当前模型：${MODEL_DISPLAY_NAME}`,
    `模型 URL：${MODEL_URL}`,
    `类别 URL：${IDX_TO_CLASS_URL}`,
    `WASM 路径：${VENDOR_URL}`,
  ];
  if (extra.sizeBytes || modelSizeBytes) lines.push(`模型大小：${formatMb(extra.sizeBytes || modelSizeBytes)}`);
  if (extra.inputNames) lines.push(`inputNames：${extra.inputNames.join(", ")}`);
  if (extra.outputNames) lines.push(`outputNames：${extra.outputNames.join(", ")}`);
  modelDetails.innerHTML = lines.map((line) => `<div>${line}</div>`).join("");
}

function setError(title, error) {
  const detail = error && (error.message || String(error)) ? `：${error.message || String(error)}` : "";
  const text = `${title}${detail}`;
  modelStatus.textContent = text;
  readyBadge.textContent = "不可用";
  readyBadge.className = "badge error";
  predLabel.textContent = text;
  confidence.textContent = "-";
  topkBars.innerHTML = "";
  setDebug(text, error && error.stack ? error.stack : error);
}

function setReady(text) {
  modelStatus.textContent = text;
  readyBadge.textContent = "模型就绪";
  readyBadge.className = "badge ready";
  setDebug(text);
}

async function fetchArrayBuffer(url, label) {
  setDebug(`${label}下载中`, { url });
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${label}文件下载失败：HTTP ${response.status}，URL=${url}`);
  }
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength < 1024) {
    throw new Error(`${label}文件异常，大小小于 1KB，可能是 Git LFS 指针文件或上传失败。URL=${url}`);
  }
  const prefix = new TextDecoder("utf-8").decode(new Uint8Array(buffer.slice(0, Math.min(96, buffer.byteLength))));
  if (prefix.startsWith("version https://git-lfs.github.com/spec/v1")) {
    throw new Error(`${label}是 Git LFS 指针文件，不是真实 ONNX 二进制。URL=${url}`);
  }
  setDebug(`${label}下载完成`, { bytes: buffer.byteLength, mb: formatMb(buffer.byteLength) });
  return buffer;
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
  setDebug("结果已显示", topk);
}

function drawSourceToSquareCanvas(source) {
  const canvas = document.createElement("canvas");
  canvas.width = IMG_SIZE;
  canvas.height = IMG_SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("无法创建 2D canvas context");
  ctx.drawImage(source, 0, 0, IMG_SIZE, IMG_SIZE);
  return canvas;
}

function preprocess(source) {
  const canvas = drawSourceToSquareCanvas(source);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const imageData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data;
  if (!imageData || imageData.length !== IMG_SIZE * IMG_SIZE * 4) {
    throw new Error("预处理失败：canvas image data empty");
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
  setDebug("图像预处理完成", { shape: [1, 3, IMG_SIZE, IMG_SIZE] });
  return input;
}

async function predictImage(source) {
  if (!session) throw new Error("模型未加载完成");
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  if (!inputName) throw new Error("ONNX input name not found");
  if (!outputName) throw new Error("ONNX output name not found");
  const input = preprocess(source);
  const tensor = new ort.Tensor("float32", input, [1, 3, IMG_SIZE, IMG_SIZE]);
  setDebug("开始 ONNX 推理", { inputName, outputName });
  const results = await session.run({ [inputName]: tensor });
  const output = results[outputName];
  if (!output || !output.data) throw new Error(`session.run 失败：output tensor not found (${outputName})`);
  const probs = softmax(Array.from(output.data));
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
    if (location.protocol === "file:") throw new Error("请通过 HTTP/HTTPS 打开，file:// 无法加载 ONNX 模型。");
    renderModelDetails();
    ort.env.wasm.wasmPaths = VENDOR_URL;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = false;
    ort.env.wasm.proxy = false;
    setDebug("ONNX Runtime WASM 设置完成", { wasmPaths: ort.env.wasm.wasmPaths });

    const classResponse = await fetch(IDX_TO_CLASS_URL, { cache: "no-store" });
    if (!classResponse.ok) throw new Error(`类别文件加载失败：HTTP ${classResponse.status}，URL=${IDX_TO_CLASS_URL}`);
    idxToClass = await classResponse.json();
    const modelBuffer = await fetchArrayBuffer(MODEL_URL, "ONNX 模型");
    modelSizeBytes = modelBuffer.byteLength;
    session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    renderModelDetails({ sizeBytes: modelSizeBytes, inputNames: session.inputNames, outputNames: session.outputNames });
    setReady(`${MODEL_DISPLAY_NAME} 已加载｜${formatMb(modelSizeBytes)}`);
  } catch (error) {
    setError("模型加载失败", error);
  }
}

async function openCamera() {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error("当前浏览器不支持 getUserMedia");
    if (stream) stream.getTracks().forEach((track) => track.stop());
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    await video.play();
    cameraPlaceholder.hidden = true;
    captureBtn.disabled = false;
    setDebug("电脑摄像头打开成功", { width: video.videoWidth, height: video.videoHeight });
  } catch (error) {
    captureBtn.disabled = true;
    setError("摄像头权限或打开失败", error);
  }
}

async function captureAndPredict() {
  try {
    if (!video.videoWidth || !video.videoHeight) throw new Error(`canvas 截图失败：video size is ${video.videoWidth}x${video.videoHeight}`);
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    const ctx = captureCanvas.getContext("2d");
    if (!ctx) throw new Error("canvas 截图失败：无法创建 2D context");
    ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    previewImage.src = captureCanvas.toDataURL("image/jpeg", 0.92);
    previewImage.style.display = "block";
    previewPlaceholder.hidden = true;
    setDebug("已从摄像头截图", { width: captureCanvas.width, height: captureCanvas.height });
    await predictImage(captureCanvas);
  } catch (error) {
    setError("拍照识别失败", error);
  }
}

function loadFileToImage(file) {
  return new Promise((resolve, reject) => {
    if (!file) return reject(new Error("未选择图片"));
    const image = new Image();
    const url = URL.createObjectURL(file);
    image.onload = () => {
      previewImage.src = url;
      previewImage.style.display = "block";
      previewPlaceholder.hidden = true;
      resolve(image);
    };
    image.onerror = () => reject(new Error("图片读取失败"));
    image.src = url;
  });
}

function resetResult() {
  predLabel.textContent = "等待识别";
  confidence.textContent = "-";
  topkBars.innerHTML = "";
  setDebug("已重置结果");
}

openCameraBtn.addEventListener("click", openCamera);
captureBtn.addEventListener("click", captureAndPredict);
resetBtn.addEventListener("click", resetResult);
localPredictBtn.addEventListener("click", async () => {
  try {
    if (!selectedImage) throw new Error("请先选择本地图片");
    await predictImage(selectedImage);
  } catch (error) {
    setError("本地图片识别失败", error);
  }
});
localFile.addEventListener("change", async () => {
  try {
    selectedImage = await loadFileToImage(localFile.files && localFile.files[0]);
    localPredictBtn.disabled = false;
    setDebug("已选择本地图片", { width: selectedImage.naturalWidth, height: selectedImage.naturalHeight });
  } catch (error) {
    selectedImage = null;
    localPredictBtn.disabled = true;
    setError("图片读取失败", error);
  }
});

loadModel();
