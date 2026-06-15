import json
from pathlib import Path

import onnx
import onnxruntime as ort


def _onnx_shape(value_info):
    dims = []
    tensor_type = value_info.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def main():
    project_dir = Path(__file__).resolve().parent
    onnx_path = project_dir / "mobile_web" / "model" / "bmw_model.onnx"
    out_dir = project_dir / "runs_deploy_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(onnx_path))
    graph_inputs = [
        {
            "name": item.name,
            "shape": _onnx_shape(item),
            "elem_type": item.type.tensor_type.elem_type,
        }
        for item in model.graph.input
    ]
    graph_outputs = [
        {
            "name": item.name,
            "shape": _onnx_shape(item),
            "elem_type": item.type.tensor_type.elem_type,
        }
        for item in model.graph.output
    ]

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = [
        {"name": item.name, "shape": item.shape, "type": item.type}
        for item in session.get_inputs()
    ]
    ort_outputs = [
        {"name": item.name, "shape": item.shape, "type": item.type}
        for item in session.get_outputs()
    ]

    info = {
        "onnx_path": str(onnx_path),
        "graph_inputs": graph_inputs,
        "graph_outputs": graph_outputs,
        "ort_inputs": ort_inputs,
        "ort_outputs": ort_outputs,
        "output_is_1x4": bool(ort_outputs and list(ort_outputs[0]["shape"]) == [1, 4]),
    }

    out_path = out_dir / "onnx_model_info.json"
    out_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if not info["output_is_1x4"]:
        raise SystemExit("ONNX output shape is not [1, 4].")


if __name__ == "__main__":
    main()
