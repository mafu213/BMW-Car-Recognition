const MODEL_VERSION = "efficientnet_b0_fix_202606";
const MODEL_DISPLAY_NAME = "EfficientNet-B0 ONNX fixed";
const MODEL_URL = new URL(`./model/bmw_model.onnx?v=${MODEL_VERSION}`, window.location.href).href;
const IDX_TO_CLASS_URL = new URL(`./model/idx_to_class.json?v=${MODEL_VERSION}`, window.location.href).href;
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
const auditFile = document.getElementById("auditFile");
const preprocessCanvas = document.getElementById("preprocessCanvas");
const predLabel = document.getElementById("predLabel");
const topkBars = document.getElementById("topkBars");

let session = null;
let idxToClass = {};
let modelSizeBytes = 0;

function safeJson(data) {
  try {
    return JSON.stringify(data, null, 2);
  } catch (error) {
    return String(data);
  }
}

function setDebug(message, data) {
  debugStatus.textContent = data === undefined ? message : `${message}\n${safeJson(data)}`;
  console.log(`[BMW Audit] ${message}`, data === undefined ? "" : data);
}

function formatMb(bytes) {
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
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

async function fetchArrayBuffer(url, label) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${label}下载失败：HTTP ${response.status}，URL=${url}`);
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength < 1024) throw new Error(`${label}文件异常，大小小于 1KB`);
  const prefix = new TextDecoder("utf-8").decode(new Uint8Array(buffer.slice(0, Math.min(96, buffer.byteLength))));
  if (prefix.startsWith("version https://git-lfs.github.com/spec/v1")) {
    throw new Error(`${label}是 Git LFS 指针文件，不是真实 ONNX 文件。`);
  }
  return buffer;
}

function drawCoverCrop(source) {
  const sourceWidth = source.naturalWidth || source.width;
  const sourceHeight = source.naturalHeight || source.height;
  if (!sourceWidth || !sourceHeight) throw new Error(`无法读取图片尺寸：${sourceWidth || 0}x${sourceHeight || 0}`);
  const ctx = preprocessCanvas.getContext("2d", { willReadFrequently: true });
  const scale = Math.max(IMG_SIZE / sourceWidth, IMG_SIZE / sourceHeight);
  const cropWidth = IMG_SIZE / scale;
  const cropHeight = IMG_SIZE / scale;
  const sx = Math.max(0, (sourceWidth - cropWidth) / 2);
  const sy = Math.max(0, (sourceHeight - cropHeight) / 2);
  ctx.clearRect(0, 0, IMG_SIZE, IMG_SIZE);
  ctx.drawImage(source, sx, sy, cropWidth, cropHeight, 0, 0, IMG_SIZE, IMG_SIZE);
  return ctx;
}

function preprocess(source) {
  const ctx = drawCoverCrop(source);
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

async function predictImage(image) {
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const input = preprocess(image);
  const tensor = new ort.Tensor("float32", input, [1, 3, IMG_SIZE, IMG_SIZE]);
  const results = await session.run({ [inputName]: tensor });
  const output = results[outputName];
  if (!output || !output.data) throw new Error(`output tensor not found: ${outputName}`);
  const logits = Array.from(output.data);
  const probs = softmax(logits);
  const classProbabilities = probs.map((prob, index) => {
    const classId = idxToClass[String(index)] || String(index);
    return { index, classId, label: displayName(classId), prob, percent: percent(prob) };
  });
  const top4 = classProbabilities.slice().sort((a, b) => b.prob - a.prob).slice(0, 4);
  predLabel.textContent = top4[0].label;
  renderTopK(top4);
  setDebug("浏览器 ONNX 推理完成", {
    modelName: MODEL_DISPLAY_NAME,
    modelVersion: MODEL_VERSION,
    modelUrl: MODEL_URL,
    modelSize: formatMb(modelSizeBytes),
    inputNames: session.inputNames,
    outputNames: session.outputNames,
    outputShape: output.dims || output.size || null,
    logits,
    softmax: classProbabilities,
    top4,
    idxToClass,
  });
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("图片读取失败"));
    image.src = URL.createObjectURL(file);
  });
}

async function loadModel() {
  try {
    ort.env.wasm.wasmPaths = VENDOR_URL;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = false;
    ort.env.wasm.proxy = false;
    const classResponse = await fetch(IDX_TO_CLASS_URL, { cache: "no-store" });
    if (!classResponse.ok) throw new Error(`类别映射加载失败：HTTP ${classResponse.status}`);
    idxToClass = await classResponse.json();
    const modelBuffer = await fetchArrayBuffer(MODEL_URL, "ONNX 模型");
    modelSizeBytes = modelBuffer.byteLength;
    session = await ort.InferenceSession.create(modelBuffer, { executionProviders: ["wasm"] });
    modelStatus.textContent = `${MODEL_DISPLAY_NAME} 已加载`;
    readyBadge.textContent = "模型就绪";
    readyBadge.className = "status ready";
    modelDetails.innerHTML = [
      `当前模型：${MODEL_DISPLAY_NAME}`,
      `模型版本：${MODEL_VERSION}`,
      `模型 URL：${MODEL_URL}`,
      `模型大小：${formatMb(modelSizeBytes)}`,
      `inputNames：${session.inputNames.join(", ")}`,
      `outputNames：${session.outputNames.join(", ")}`,
      `idx_to_class：${safeJson(idxToClass)}`,
    ].map((line) => `<div>${line}</div>`).join("");
    setDebug("模型加载完成");
  } catch (error) {
    modelStatus.textContent = `模型加载失败：${error.message || error}`;
    readyBadge.textContent = "不可用";
    readyBadge.className = "status error";
    setDebug("模型加载失败", error.stack || error.message || String(error));
  }
}

auditFile.addEventListener("change", async () => {
  try {
    const file = auditFile.files && auditFile.files[0];
    if (!file) return;
    if (!session) throw new Error("模型未加载完成");
    const image = await loadImage(file);
    await predictImage(image);
  } catch (error) {
    predLabel.textContent = `推理失败：${error.message || error}`;
    setDebug("推理失败", error.stack || error.message || String(error));
  }
});

loadModel();
