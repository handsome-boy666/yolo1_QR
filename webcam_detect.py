import os
import time
import argparse
import yaml
import torch
import torchvision.transforms as transforms

def _sanitize_path(p: str | None) -> str:
    if not p:
        return ""
    s = str(p).strip()
    if (s.startswith('r"') and s.endswith('"')) or (s.startswith("r'") and s.endswith("'")):
        s = s[2:-1]
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    s = s.replace("\\", "/")
    return os.path.normpath(s)

def read_predict_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    pcfg = cfg.get("predict", {})
    return {
        "ckpt_path": _sanitize_path(pcfg.get("ckpt_path") or ""),
        "img_size": int(pcfg.get("img_size", 512)),
        "S": int(pcfg.get("S", 4)),
        "conf_thresh": float(pcfg.get("conf_thresh", 0.5)),
        "iou_thresh": float(pcfg.get("iou_thresh", 0.5)),
        "train_log_dir": _sanitize_path((cfg.get("train", {}) or {}).get("log_dir", "./logs")),
        "live_save_dir": _sanitize_path(pcfg.get("live_save_dir", "./predictions/live")),
        "webcam_index": int(pcfg.get("webcam_index", 0)),
        "record": bool(pcfg.get("record", False)),
    }

def find_latest_checkpoint(search_dir: str) -> str | None:
    latest = None
    latest_mtime = -1.0
    if not os.path.isdir(search_dir):
        return None
    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.endswith(".pth"):
                path = os.path.join(root, f)
                try:
                    mtime = os.path.getmtime(path)
                except Exception:
                    mtime = 0.0
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest = path
    return latest

from models.yolov1QR import YOLOv1_QR
from utils.utils import cell_to_bbox
from models.utils import nms

def load_model(ckpt_path: str | None, device: torch.device) -> YOLOv1_QR:
    model = YOLOv1_QR().to(device)
    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model_state", state))
    return model

def build_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

def decode_boxes(pred: torch.Tensor, S: int, conf_thresh: float, iou_thresh: float):
    B = pred.size(0)
    pred = pred.view(B, S, S, 5).detach().cpu()
    boxes = []
    scores = []
    for j in range(S):
        for i in range(S):
            conf = float(pred[0, j, i, 4])
            if conf >= conf_thresh:
                x1, y1, x2, y2 = cell_to_bbox(
                    i, j,
                    float(pred[0, j, i, 0]),
                    float(pred[0, j, i, 1]),
                    float(pred[0, j, i, 2]),
                    float(pred[0, j, i, 3]), S)
                boxes.append((x1, y1, x2, y2))
                scores.append(conf)
    keep = nms(boxes, scores, iou_thresh)
    return [(boxes[k], scores[k]) for k in keep]

def main():
    parser = argparse.ArgumentParser(description="Webcam QR Real-time Detection")
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--index", type=int, default=None)  # 摄像头索引，默认 0
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    cfg = read_predict_config(args.config)
    if args.index is not None:
        cfg["webcam_index"] = int(args.index)
    if args.record:
        cfg["record"] = True

    try:
        import cv2
    except Exception:
        print("未找到 OpenCV (cv2)。请在 Anaconda 环境中安装: pip install opencv-python")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = cfg["ckpt_path"] or find_latest_checkpoint(cfg["train_log_dir"]) or ""
    if not ckpt_path:
        print("未找到权重文件，请在 predict.ckpt_path 指定或训练后再试")
        return
    model = load_model(ckpt_path, device)
    model.eval()
    tfm = build_transform(cfg["img_size"])

    cap = cv2.VideoCapture(cfg["webcam_index"])
    if not cap.isOpened():
        print(f"无法打开摄像头: index={cfg['webcam_index']}")
        return

    os.makedirs(cfg["live_save_dir"], exist_ok=True)
    writer = None
    out_path = None

    def start_record(h_w):
        nonlocal writer, out_path
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = os.path.join(cfg["live_save_dir"], time.strftime("%Y%m%d_%H%M%S") + ".mp4")
            writer = cv2.VideoWriter(out_path, fourcc, 20.0, h_w)
            print(f"开始录制: {out_path}")

    def stop_record():
        nonlocal writer, out_path
        if writer is not None:
            writer.release()
            print(f"录制已保存: {out_path}")
            writer = None
            out_path = None

    # 预读一帧以获取尺寸
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头帧")
        cap.release()
        return
    h, w = frame.shape[:2]
    if cfg["record"]:
        start_record((w, h))

    try:
        while True:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            import PIL.Image as Image
            img = Image.fromarray(rgb)
            img_t = tfm(img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(img_t)
            boxes = decode_boxes(pred, cfg["S"], cfg["conf_thresh"], cfg["iou_thresh"]) 
            for (x1, y1, x2, y2), score in boxes:
                x1p = int(x1 * w)
                y1p = int(y1 * h)
                x2p = int(x2 * w)
                y2p = int(y2 * h)
                cv2.rectangle(frame, (x1p, y1p), (x2p, y2p), (0, 0, 255), 2)
                label = f"{score:.2f}"
                ty = y1p - 12 if (y1p - 12) > 0 else (y1p + 2)
                tx = x1p + 2
                cv2.rectangle(frame, (tx - 1, ty - 14), (tx + 60, ty + 2), (0, 0, 0), -1)
                cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("YOLOv1_QR Live", frame)
            if writer is not None:
                writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                snap = os.path.join(cfg["live_save_dir"], time.strftime("%Y%m%d_%H%M%S") + ".jpg")
                cv2.imwrite(snap, frame)
                print(f"已保存快照: {snap}")
            if key == ord('r'):
                if writer is None:
                    start_record((w, h))
                else:
                    stop_record()

            ret, frame = cap.read()
            if not ret:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        stop_record()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass

if __name__ == "__main__":
    main()
