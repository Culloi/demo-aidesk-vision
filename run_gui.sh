#!/usr/bin/env bash
# RX-78 桌面陪伴机器人启动脚本（主机 / 开发板通用）
#   ./run_gui.sh            普通窗口
#   ./run_gui.sh --robot    全屏模式
#   ./run_gui.sh --lang en  指定语言
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "[ERROR] .venv 不存在，请先按 README.md 创建虚拟环境。"
  exit 1
fi

# 从 SSH 或开机自启动运行时没有显示环境变量：自动连接本机已登录用户的桌面会话
if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ]; then
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  if [ -S "$XDG_RUNTIME_DIR/wayland-0" ]; then
    export WAYLAND_DISPLAY=wayland-0
  elif [ -S /tmp/.X11-unix/X0 ]; then
    export DISPLAY=:0
  fi
fi

# Wayland 桌面（如开发板的 GNOME）优先用 Wayland，失败时退回 X11（Xwayland）
if [ -n "$WAYLAND_DISPLAY" ] && [ -z "$QT_QPA_PLATFORM" ]; then
  export QT_QPA_PLATFORM="wayland;xcb"
fi

source .venv/bin/activate
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m usb_person_detector.entrypoints.gui_app "$@"
