#!/usr/bin/env bash
# RX-78 桌面陪伴机器人环境部署脚本（Debian/Ubuntu、x86_64/aarch64）
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_INDEX_URL="${PIP_INDEX_URL:-}"
INSTALL_SYSTEM=1
CHECK_ONLY=0

usage() {
  cat <<'EOF'
用法：./setup_env.sh [选项]

选项：
  --skip-system       不安装 apt 系统依赖（无 sudo 权限时使用）
  --china-mirror      使用阿里云 PyPI 镜像
  --check-only        不安装，仅执行环境自检和单元测试
  --python PATH       指定 Python 解释器（默认 python3）
  -h, --help          显示帮助

环境变量：
  PYTHON_BIN          Python 解释器，效果同 --python
  PIP_INDEX_URL       自定义 PyPI 镜像；命令行 --china-mirror 优先
EOF
}

while (($#)); do
  case "$1" in
    --skip-system) INSTALL_SYSTEM=0 ;;
    --china-mirror) PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/" ;;
    --check-only) CHECK_ONLY=1 ;;
    --python)
      shift
      [[ $# -gt 0 ]] || { echo "[错误] --python 缺少解释器路径" >&2; exit 2; }
      PYTHON_BIN="$1"
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[错误] 未知选项：$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

cd "$ROOT_DIR"

check_models() {
  "$VENV_DIR/bin/python" - <<'PY'
import json
from pathlib import Path

root = Path.cwd()
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
required = [(config.get("model", "models/yolov8n.onnx"), "主视觉检测")]
if config.get("fire_enabled", False):
    required.append((config.get("fire_model", "models/fire_yolov8n.onnx"), "明火检测"))

missing = []
for relative, purpose in required:
    path = root / relative
    if path.is_file():
        print(f"  [通过] {purpose}模型：{relative}")
    else:
        missing.append(f"{relative}（{purpose}）")

asr_dir = root / config.get(
    "asr_model_dir",
    "models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
)
if (asr_dir / "model.int8.onnx").is_file() and (asr_dir / "tokens.txt").is_file():
    print(f"  [通过] 语音识别模型：{asr_dir.relative_to(root)}")
else:
    print("  [提示] SenseVoice 模型不完整，语音指令功能将自动禁用")

if missing:
    raise SystemExit("[错误] 缺少当前配置需要的模型：\n  " + "\n  ".join(missing)
                     + "\n模型用途和目录见 DEPLOYMENT.md。")
PY
}

if [[ $CHECK_ONLY -eq 0 ]]; then
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "[错误] 找不到 Python：$PYTHON_BIN" >&2
    exit 1
  }

  if [[ $INSTALL_SYSTEM -eq 1 ]]; then
    if ! command -v apt-get >/dev/null 2>&1; then
      echo "[提示] 当前系统没有 apt-get，请按 DEPLOYMENT.md 手动安装系统依赖。"
    else
      APT=(apt-get)
      if [[ $(id -u) -ne 0 ]]; then
        command -v sudo >/dev/null 2>&1 || {
          echo "[错误] 安装系统依赖需要 root 或 sudo；可使用 --skip-system 跳过。" >&2
          exit 1
        }
        APT=(sudo apt-get)
      fi
      echo "[1/5] 安装系统依赖..."
      "${APT[@]}" update
      "${APT[@]}" install -y \
        python3-venv python3-pip fonts-noto-cjk alsa-utils \
        libgl1 libglib2.0-0 libxcb-cursor0 libxkbcommon-x11-0 \
        libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-render-util0 libxcb-xinerama0
    fi
  else
    echo "[1/5] 已跳过系统依赖安装。"
  fi

  echo "[2/5] 创建/复用虚拟环境：$VENV_DIR"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi

  echo "[3/5] 安装 Python 依赖..."
  PIP_ARGS=(--disable-pip-version-check)
  EFFECTIVE_INDEX="$PIP_INDEX_URL"
  if [[ -n "$EFFECTIVE_INDEX" ]]; then
    PIP_ARGS+=(--index-url "$EFFECTIVE_INDEX")
  fi
  "$VENV_DIR/bin/python" -m pip install "${PIP_ARGS[@]}" --upgrade pip setuptools wheel
  "$VENV_DIR/bin/python" -m pip install "${PIP_ARGS[@]}" -r requirements.txt

  echo "[4/5] 检查运行时模型..."
  mkdir -p models captures
else
  echo "[1-4/5] 仅检查模式，不执行安装。"
  [[ -x "$VENV_DIR/bin/python" ]] || {
    echo "[错误] 虚拟环境不存在，请先运行 ./setup_env.sh" >&2
    exit 1
  }
fi

check_models

echo "[5/5] 执行环境自检和单元测试..."
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
"$VENV_DIR/bin/python" -m usb_person_detector.entrypoints.env_check
"$VENV_DIR/bin/python" -m unittest discover -s tests

cat <<EOF

[完成] 项目环境部署成功。
启动命令：
  cd "$ROOT_DIR"
  ./run_gui.sh

全屏机器人模式：
  ./run_gui.sh --robot
EOF
