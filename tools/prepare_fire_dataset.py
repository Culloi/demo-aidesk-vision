#!/usr/bin/env python3
"""D-Fire 数据集 -> 单类（fire）YOLO 数据集。

D-Fire 原始标注里：0 = fire，1 = smoke。本项目只做明火检测，
所以保留 fire（重映射为 0）、删除 smoke 行；没有标注框的图片（纯负样本）保留，
用来压低误报——这是明火检测能不能用的关键。

用法：
    python3 tools/prepare_fire_dataset.py --src ~/dfire --dst ~/dfire_fire
    python3 tools/prepare_fire_dataset.py --src ~/dfire --dst ~/dfire_fire --dry-run

生成结构（和 Ultralytics 约定一致）：
    <dst>/
    ├── images/{train,val,test}/*.jpg
    ├── labels/{train,val,test}/*.txt
    └── fire.yaml
"""
import argparse
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def detect_fire_class(src: Path) -> tuple:
    """从 data.yaml 读取类别名，确定 fire / smoke 的编号。

    不同 D-Fire 分发版的类别顺序不一致：
      - D-Fire 原始版：0=fire, 1=smoke
      - Kaggle sayedgamal99 版：0=smoke, 1=fire
    所以不能写死，优先读 data.yaml 的 names 列表。

    返回 (fire_id, smoke_id)；无法判定时返回 (None, None) 交给调用方报错。
    """
    for cand in (src / "data.yaml", src.parent / "data.yaml", src / "fire.yaml"):
        if not cand.exists():
            continue
        text = cand.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"names\s*:\s*(\[[^\]]*\])", text)
        if not m:
            # 形如 names:\n  0: fire\n  1: smoke
            items = re.findall(r"^\s*(\d+)\s*:\s*['\"]?([A-Za-z_]+)['\"]?", text, re.M | re.M)
            if items:
                names = {int(i): n.lower() for i, n in items}
            else:
                continue
        else:
            raw = [x.strip().strip("'\"[]").strip() for x in m.group(1).split(",") if x.strip()]
            names = {i: n.lower() for i, n in enumerate(raw)}
        fire_id = next((i for i, n in names.items() if "fire" in n), None)
        smoke_id = next((i for i, n in names.items() if "smoke" in n), None)
        if fire_id is not None:
            print(f"从 {cand} 读取类别映射：{names} -> fire={fire_id}, smoke={smoke_id}")
            return fire_id, smoke_id
    return None, None


def find_split_dirs(src: Path):
    """在 src 下查找 (images_dir, labels_dir, split_name) 三元组。

    兼容常见布局：
        src/images/{train,val,test}  +  src/labels/{train,val,test}
        src/{train,val,test}/images  +  src/{train,val,test}/labels
        src/train/images             +  src/train/labels
    """
    found = []
    # 布局 A：images/<split> 与 labels/<split>
    img_root = src / "images"
    if img_root.is_dir():
        for split_dir in sorted(p for p in img_root.iterdir() if p.is_dir()):
            lab_dir = src / "labels" / split_dir.name
            if lab_dir.is_dir():
                found.append((split_dir, lab_dir, split_dir.name))
    # 布局 B：<split>/images 与 <split>/labels
    if not found:
        for split_dir in sorted(p for p in src.iterdir() if p.is_dir()):
            if (split_dir / "images").is_dir() and (split_dir / "labels").is_dir():
                found.append((split_dir / "images", split_dir / "labels", split_dir.name))
    return found


def filter_label_file(label_path: Path, fire_id: int):
    """返回 (保留的 fire 行列表, 删除的非 fire 行数, 原始行数)。"""
    kept, dropped = [], 0
    text = label_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        if cls != fire_id:
            dropped += 1
            continue
        parts[0] = "0"                 # 统一重映射为 0（单类模型）
        kept.append(" ".join(parts))
    return kept, dropped, len(text.splitlines())


def main():
    p = argparse.ArgumentParser(description="D-Fire -> 单类 fire 数据集")
    p.add_argument("--src", required=True, help="D-Fire 解压后的根目录")
    p.add_argument("--dst", required=True, help="输出目录（会新建 images/labels）")
    p.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    p.add_argument("--fire-class", type=int, default=None,
                   help="fire 的类别编号；默认从 data.yaml 自动识别（不同分发版 0/1 不同）")
    p.add_argument("--smoke-class", type=int, default=None, help="smoke 的类别编号，仅用于日志")
    args = p.parse_args()

    src, dst = Path(args.src).expanduser(), Path(args.dst).expanduser()
    if not src.is_dir():
        print(f"源目录不存在：{src}", file=sys.stderr)
        sys.exit(1)

    splits = find_split_dirs(src)
    if not splits:
        print("没有找到 images/labels 结构，请确认 --src 指向 D-Fire 解压根目录", file=sys.stderr)
        print("已支持布局：images/<split> | <split>/images | train/images", file=sys.stderr)
        sys.exit(1)

    print(f"源目录：{src}")
    print(f"输出目录：{dst}{'（dry-run，不写入）' if args.dry_run else ''}\n")

    # 确定哪一个类别编号是 fire（不同 D-Fire 分发版顺序不同，必须不能搞错）
    fire_id, smoke_id = detect_fire_class(src)
    if args.fire_class is not None:
        fire_id = args.fire_class
        print(f"手动指定 fire 类别编号：{fire_id}")
    if args.smoke_class is not None:
        smoke_id = args.smoke_class
    if fire_id is None:
        print("无法自动判断哪个类别是 fire。请在 data.yaml 里写明 names，"
              "或用 --fire-class 手动指定（本项目 Kaggle 版是 fire=1）", file=sys.stderr)
        sys.exit(1)
    print(f"保留类别 {fire_id}（fire）→ 重映射为 0"
          + (f"；删除类别 {smoke_id}（smoke）" if smoke_id is not None else ""))
    print()

    total_kept_box = total_dropped = total_img = 0
    for img_dir, lab_dir, split in splits:
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        out_img = dst / "images" / split
        out_lab = dst / "labels" / split
        if not args.dry_run:
            out_img.mkdir(parents=True, exist_ok=True)
            out_lab.mkdir(parents=True, exist_ok=True)

        stat = Counter()
        for img in images:
            label = lab_dir / (img.stem + ".txt")
            if not label.exists():
                stat["no_label"] += 1
                continue
            kept, dropped, _ = filter_label_file(label, fire_id)
            stat["boxes_kept"] += len(kept)
            stat["boxes_dropped"] += dropped
            if kept:
                stat["img_with_fire"] += 1
            else:
                stat["img_negative"] += 1
            if not args.dry_run:
                # 写过滤后的标注（可能是空文件，空文件 = 纯负样本，必须保留）
                (out_lab / label.name).write_text(
                    "\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
                shutil.copy2(img, out_img / img.name)

        total_kept_box += stat["boxes_kept"]
        total_dropped += stat["boxes_dropped"]
        total_img += len(images) - stat["no_label"]
        print(f"[{split}] 图片 {len(images)} 张"
              f" | 有火 {stat['img_with_fire']} / 负样本 {stat['img_negative']}"
              f" | fire 框 {stat['boxes_kept']} | 删除 smoke 框 {stat['boxes_dropped']}"
              + (f" | 缺标注跳过 {stat['no_label']}" if stat["no_label"] else ""))

    print(f"\n合计：图片 {total_img} 张，fire 框 {total_kept_box} 个，删除 smoke 框 {total_dropped} 个")

    if args.dry_run:
        print("\n（dry-run 结束，未写入任何文件）")
        return

    yaml_path = dst / "fire.yaml"
    split_names = [s for _, _, s in splits]
    lines = [f"path: {dst.resolve()}", "train: images/train", "val: images/val"]
    if "test" in split_names:
        lines.append("test: images/test")
    lines += ["names:", "  0: fire", ""]
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已生成 {yaml_path}")
    print("训练命令示例：yolo detect train model=yolov8n.pt "
          f"data={yaml_path} imgsz=640 epochs=80 batch=32 name=fire_yolov8n")


if __name__ == "__main__":
    main()
