const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const emptyPreview = document.getElementById("emptyPreview");
const predictButton = document.getElementById("predictButton");
const className = document.getElementById("className");
const confidenceValue = document.getElementById("confidenceValue");
const bars = document.getElementById("bars");

let selectedFile = null;

function setBusy(isBusy) {
  predictButton.disabled = isBusy || !selectedFile;
  predictButton.textContent = isBusy ? "识别中..." : "开始识别";
}

function percent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function renderBars(items) {
  bars.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "bar-row";

    const head = document.createElement("div");
    head.className = "bar-head";

    const label = document.createElement("span");
    label.className = "bar-label";
    label.textContent = item.display_name;

    const value = document.createElement("span");
    value.textContent = percent(item.probability);

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = percent(item.probability);

    head.append(label, value);
    track.append(fill);
    row.append(head, track);
    bars.append(row);
  });
}

imageInput.addEventListener("change", () => {
  selectedFile = imageInput.files && imageInput.files[0] ? imageInput.files[0] : null;
  if (!selectedFile) {
    previewImage.style.display = "none";
    emptyPreview.style.display = "grid";
    setBusy(false);
    return;
  }

  const url = URL.createObjectURL(selectedFile);
  previewImage.onload = () => URL.revokeObjectURL(url);
  previewImage.src = url;
  previewImage.style.display = "block";
  emptyPreview.style.display = "none";
  className.textContent = "-";
  confidenceValue.textContent = "-";
  bars.innerHTML = "";
  setBusy(false);
});

predictButton.addEventListener("click", async () => {
  if (!selectedFile) return;
  setBusy(true);

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "识别失败");
    }

    className.textContent = data.display_name;
    confidenceValue.textContent = percent(data.confidence);
    renderBars(data.top4);
  } catch (error) {
    className.textContent = "识别失败";
    confidenceValue.textContent = "-";
    bars.innerHTML = "";
    alert(error.message);
  } finally {
    setBusy(false);
  }
});
