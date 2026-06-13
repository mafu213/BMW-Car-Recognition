const classNameEl = document.getElementById("className");
const confidenceValueEl = document.getElementById("confidenceValue");
const barsEl = document.getElementById("bars");

const cameraVideo = document.getElementById("cameraVideo");
const captureCanvas = document.getElementById("captureCanvas");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const cameraHint = document.getElementById("cameraHint");
const openCameraButton = document.getElementById("openCameraButton");
const captureButton = document.getElementById("captureButton");

const cameraUploadInput = document.getElementById("cameraUploadInput");
const cameraUploadButton = document.getElementById("cameraUploadButton");
const cameraUploadPreview = document.getElementById("cameraUploadPreview");
const cameraUploadEmpty = document.getElementById("cameraUploadEmpty");

const localImageInput = document.getElementById("localImageInput");
const localUploadButton = document.getElementById("localUploadButton");
const localPreview = document.getElementById("localPreview");
const localEmpty = document.getElementById("localEmpty");

let cameraStream = null;
let cameraUploadFile = null;
let localImageFile = null;

function percent(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function setButtonBusy(button, busy, textWhenIdle) {
  button.disabled = busy;
  button.textContent = busy ? "识别中..." : textWhenIdle;
}

function setStatusMessage(message) {
  classNameEl.textContent = message;
  confidenceValueEl.textContent = "-";
  barsEl.innerHTML = "";
}

function normalizeTopK(data) {
  if (Array.isArray(data.topk)) {
    return data.topk.map((item) => ({
      label: item.label,
      prob: item.prob,
    }));
  }
  if (Array.isArray(data.top4)) {
    return data.top4.map((item) => ({
      label: item.display_name || item.class_name,
      prob: item.probability,
    }));
  }
  return [];
}

function renderTopK(items) {
  barsEl.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "bar-row";

    const head = document.createElement("div");
    head.className = "bar-head";

    const label = document.createElement("span");
    label.className = "bar-label";
    label.textContent = item.label;

    const value = document.createElement("span");
    value.className = "bar-value";
    value.textContent = percent(item.prob);

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = percent(item.prob);

    head.append(label, value);
    track.append(fill);
    row.append(head, track);
    barsEl.append(row);
  });
}

async function uploadAndPredict(blobOrFile, filename, busyButton, idleText) {
  if (!blobOrFile) {
    return;
  }

  if (busyButton) {
    setButtonBusy(busyButton, true, idleText);
  }
  setStatusMessage("正在识别...");

  const formData = new FormData();
  formData.append("file", blobOrFile, filename || "bmw_capture.jpg");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "识别失败");
    }

    classNameEl.textContent = data.pred_label || data.display_name || data.predicted_class || "-";
    confidenceValueEl.textContent = percent(data.confidence || 0);
    renderTopK(normalizeTopK(data));
  } catch (error) {
    setStatusMessage("识别失败");
    alert(error.message || "识别失败，请重新拍照或上传图片。");
  } finally {
    if (busyButton) {
      setButtonBusy(busyButton, false, idleText);
    }
    refreshButtons();
  }
}

async function openCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showCameraFallback();
    return;
  }

  try {
    if (cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop());
    }
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false,
    });
    cameraVideo.srcObject = cameraStream;
    cameraPlaceholder.hidden = true;
    cameraHint.hidden = true;
    captureButton.disabled = false;
    setStatusMessage("摄像头已打开，请对准图片后拍照识别");
  } catch (error) {
    console.warn("Camera open failed:", error);
    showCameraFallback();
  }
}

function showCameraFallback() {
  cameraHint.hidden = false;
  cameraPlaceholder.hidden = false;
  captureButton.disabled = true;
  setStatusMessage("请使用拍照上传备用区");
}

function captureAndPredict() {
  if (!cameraVideo.videoWidth || !cameraVideo.videoHeight) {
    showCameraFallback();
    return;
  }

  captureCanvas.width = cameraVideo.videoWidth;
  captureCanvas.height = cameraVideo.videoHeight;
  const context = captureCanvas.getContext("2d");
  context.drawImage(cameraVideo, 0, 0, captureCanvas.width, captureCanvas.height);

  captureCanvas.toBlob(
    (blob) => {
      if (!blob) {
        alert("拍照失败，请使用拍照上传备用区。");
        return;
      }
      uploadAndPredict(blob, "camera_capture.jpg", captureButton, "拍照识别");
    },
    "image/jpeg",
    0.92,
  );
}

function previewFile(file, imageEl, emptyEl) {
  if (!file) {
    imageEl.removeAttribute("src");
    imageEl.style.display = "none";
    emptyEl.style.display = "grid";
    return;
  }

  const url = URL.createObjectURL(file);
  imageEl.onload = () => URL.revokeObjectURL(url);
  imageEl.src = url;
  imageEl.style.display = "block";
  emptyEl.style.display = "none";
}

function refreshButtons() {
  cameraUploadButton.disabled = !cameraUploadFile;
  localUploadButton.disabled = !localImageFile;
  if (cameraStream) {
    captureButton.disabled = false;
  }
}

openCameraButton.addEventListener("click", openCamera);
captureButton.addEventListener("click", captureAndPredict);

cameraUploadInput.addEventListener("change", () => {
  cameraUploadFile = cameraUploadInput.files && cameraUploadInput.files[0] ? cameraUploadInput.files[0] : null;
  previewFile(cameraUploadFile, cameraUploadPreview, cameraUploadEmpty);
  setStatusMessage(cameraUploadFile ? "照片已选择，点击识别照片" : "等待识别");
  refreshButtons();
});

cameraUploadButton.addEventListener("click", () => {
  uploadAndPredict(cameraUploadFile, cameraUploadFile?.name || "iphone_capture.jpg", cameraUploadButton, "识别照片");
});

localImageInput.addEventListener("change", () => {
  localImageFile = localImageInput.files && localImageInput.files[0] ? localImageInput.files[0] : null;
  previewFile(localImageFile, localPreview, localEmpty);
  setStatusMessage(localImageFile ? "图片已选择，点击识别图片" : "等待识别");
  refreshButtons();
});

localUploadButton.addEventListener("click", () => {
  uploadAndPredict(localImageFile, localImageFile?.name || "local_image.jpg", localUploadButton, "识别图片");
});

if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
  cameraHint.hidden = false;
}
