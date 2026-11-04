#!/usr/bin/env python3
"""ONNX → RKNN 模型转换（在 x86 主机上运行，需要单独的 rknn-toolkit2 虚拟环境，安装方法见 tools/requirements-rknn.txt）。

  python tools/convert_rknn.py                       # FP16，不需要校准图片，精度与 ONNX 基本一致
  python tools/convert_rknn.py --quant int8 --calib captures/   # INT8 量化，需要 50~200 张真实场景图片

转换完成后会用 RKNN 模拟器在测试图片上推理，与 ONNX Runtime 的检测结果对比，不上板子就能知道转得对不对。

RKNN 模型的输入约定（板端推理时要按这个格式送数据）：
  NHWC、uint8、RGB、640x640 letterbox（与 detector.letterbox 相同），归一化（/255）已经写进模型。
输出与 ONNX 相同：[1, 84, 8400]，可以直接复用 detector.postprocess。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from usb_person_detector.vision.detector import DEFAULT_CLASSES, YoloDetector  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description="ONNX → RKNN 模型转换")
    p.add_argument("--onnx", default=str(ROOT / "models/yolov8n.onnx"))
    p.add_argument("--target", default="rk3576", help="目标芯片，默认 rk3576")
    p.add_argument("--quant", choices=["fp16", "int8"], default="fp16")
    p.add_argument("--calib", default=None, help="INT8 校准图片目录（递归查找 jpg/png）")
    p.add_argument("--calib-max", type=int, default=200, help="最多使用多少张校准图片")
    p.add_argument("--out", default=None, help="输出路径，默认 models/yolov8n_<target>_<quant>.rknn")
    p.add_argument("--verify", nargs="*", default=None,
                   help="用于对比的测试图片，默认使用 ultralytics 自带的 bus.jpg / zidane.jpg")
    p.add_argument("--classes", default=None,
                   help="模型类别，逗号分隔（如 fire 或 person,cat,dog）；默认按 COCO 类别名显示")
    p.add_argument("--class-conf", type=float, default=None, help="验证时使用的置信度阈值")
    return p.parse_args()


def find_images(folder: Path, limit: int):
    images = sorted(f for f in folder.rglob("*") if f.suffix.lower() in IMAGE_EXT)
    if len(images) > limit:  # 均匀抽样，避免全是连续帧
        idx = np.linspace(0, len(images) - 1, limit).astype(int)
        images = [images[i] for i in idx]
    return images


def default_verify_images():
    found = []
    for name in ("bus.jpg", "zidane.jpg"):
        found += list(ROOT.glob(f".venv*/lib/python3*/site-packages/ultralytics/assets/{name}"))[:1]
    found += find_images(ROOT / "captures", 3)
    return found


def iou(a, b):
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def compare(ref, got):
    """按类别 + IoU 匹配两组检测框，返回 (匹配数, 平均置信度差, 平均 IoU)。"""
    used, diffs, ious = set(), [], []
    for r in ref:
        best, best_iou = None, 0.5
        for j, g in enumerate(got):
            if j in used or g.cls_id != r.cls_id:
                continue
            v = iou(r, g)
            if v > best_iou:
                best, best_iou = j, v
        if best is not None:
            used.add(best)
            diffs.append(abs(got[best].score - r.score))
            ious.append(best_iou)
    return len(diffs), (float(np.mean(diffs)) if diffs else 0.0), (float(np.mean(ious)) if ious else 0.0)


def main():
    args = parse_args()
    try:
        from rknn.api import RKNN
    except ImportError:
        sys.exit("[ERROR] 没有找到 rknn-toolkit2，请先激活转换环境：source .venv-rknn/bin/activate")

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        sys.exit(f"[ERROR] 找不到 ONNX 模型：{onnx_path}")
    out = Path(args.out or ROOT / f"models/{onnx_path.stem}_{args.target}_{args.quant}.rknn")

    dataset = None
    if args.quant == "int8":
        if not args.calib:
            sys.exit("[ERROR] INT8 量化需要校准图片：--calib <图片目录>（建议 50~200 张真实使用场景的照片）")
        images = find_images(Path(args.calib), args.calib_max)
        if len(images) < 20:
            sys.exit(f"[ERROR] 校准图片只有 {len(images)} 张，至少需要 20 张，建议 50~200 张")
        dataset = out.with_suffix(".dataset.txt")
        dataset.write_text("\n".join(str(p.resolve()) for p in images) + "\n")
        print(f"校准图片 {len(images)} 张 → {dataset}")

    rknn = RKNN(verbose=False)
    # 归一化写进模型：板端直接送 uint8 RGB，省掉 CPU 上的 /255 和类型转换
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=args.target)

    print(f"加载 {onnx_path} ...")
    if rknn.load_onnx(model=str(onnx_path)) != 0:
        sys.exit("[ERROR] load_onnx 失败")
    print(f"构建 {args.quant.upper()} 模型（目标 {args.target}）...")
    t = time.time()
    if rknn.build(do_quantization=args.quant == "int8", dataset=str(dataset) if dataset else None) != 0:
        sys.exit("[ERROR] build 失败")
    print(f"构建完成，用时 {time.time() - t:.0f} 秒")
    if rknn.export_rknn(str(out)) != 0:
        sys.exit("[ERROR] export_rknn 失败")
    print(f"已导出 {out}（{out.stat().st_size / 1048576:.1f} MB）")

    # ---------- 用模拟器验证 ----------
    verify = [Path(p) for p in args.verify] if args.verify else default_verify_images()
    if not verify:
        print("没有找到测试图片，跳过验证（可用 --verify a.jpg b.jpg 指定）")
        rknn.release()
        return
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if args.classes:
        names = [s.strip() for s in args.classes.split(",") if s.strip()]
        cls_map = {i: n for i, n in enumerate(names)}
    else:
        cls_map = DEFAULT_CLASSES
    det = YoloDetector(str(onnx_path), args.class_conf or config.get("conf_thres", 0.4),
                       config.get("iou_thres", 0.5),
                       classes=cls_map, class_conf=config.get("class_conf"))
    print("\n启动 RKNN 模拟器做精度对比 ...")
    if rknn.init_runtime() != 0:
        sys.exit("[ERROR] 模拟器启动失败")

    total_ref = total_match = 0
    for path in verify:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  跳过（无法读取）：{path}")
            continue
        ref = det.infer(img)
        padded, scale, pad_x, pad_y = det.letterbox(img)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        outputs = rknn.inference(inputs=[rgb[None, ...]], data_format="nhwc")
        got = det.postprocess(outputs[0], img.shape[:2], scale, pad_x, pad_y)
        matched, score_diff, mean_iou = compare(ref, got)
        total_ref += len(ref)
        total_match += matched
        print(f"  {path.name}: ONNX {len(ref)} 个 / RKNN {len(got)} 个，匹配 {matched} 个，"
              f"平均置信度差 {score_diff:.3f}，平均 IoU {mean_iou:.2f}")
        print(f"      ONNX: {sorted((d.label, round(d.score, 2)) for d in ref)}")
        print(f"      RKNN: {sorted((d.label, round(d.score, 2)) for d in got)}")
    rknn.release()

    rate = total_match / total_ref if total_ref else 1.0
    print(f"\n总计：ONNX 检测到 {total_ref} 个目标，RKNN 匹配上 {total_match} 个（{rate:.0%}）")
    print("结论：" + ("转换正常，可以上板测试" if rate >= 0.9 else
                    "差异较大，请把以上输出发给我分析（INT8 可先改用 FP16）"))


if __name__ == "__main__":
    main()
