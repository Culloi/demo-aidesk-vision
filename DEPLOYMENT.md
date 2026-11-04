# RX-78 桌面陪伴机器人环境部署文档

本文适用于两类环境：

- x86_64 Debian/Ubuntu 开发机：使用 ONNX Runtime 在 CPU 上推理。
- Quectel PI M2（RK3576、aarch64、Debian 13）：优先使用 RKNN NPU，失败时自动回退到 ONNX CPU。

## 1. 硬件与软件要求

| 项目 | 最低要求 | 说明 |
|---|---|---|
| 操作系统 | Debian/Ubuntu Linux | GUI 为 PySide6 本地应用，不是 Web 服务 |
| Python | 3.10+ | 已验证开发机 Python 3.10、开发板 Python 3.13 |
| 摄像头 | UVC USB 摄像头 | 推荐支持 1280×720 MJPG；无摄像头也可完成环境自检和测试 |
| 显示 | X11 或 Wayland | RK3576 开发板使用 GNOME Wayland |
| 麦克风 | ALSA 可识别设备（可选） | 仅语音指令需要，录音程序为 `arecord` |
| NPU | RK3576（可选） | 仅 `.rknn` 模型需要；系统需提供 `/usr/lib/librknnrt.so` |

## 2. 模型目录、文件及作用

所有模型统一放在项目根目录的 `models/` 下。相对路径必须与 `config.json` 一致。

| 模型文件/目录 | 是否必需 | 运行平台 | 作用 |
|---|---|---|---|
| `models/yolov8n.onnx` | **必需** | x86_64 / aarch64 CPU | 主视觉检测模型（COCO）：检测人、手机、猫、狗；驱动有人/无人、目光跟随、玩手机提醒和宠物识别。NPU 不可用时也用它回退到 CPU |
| `models/yolov8n_rk3576_fp16.rknn` | RK3576 推荐 | RK3576 NPU | 上述主视觉模型的 RKNN FP16 版本，用 NPU 提升实时帧率；由 `tools/convert_rknn.py` 从 ONNX 转换 |
| `models/fire_yolov8n.onnx` | 开启明火检测时必需 | x86_64 / aarch64 CPU | 单类别明火检测模型；`config.json` 中 `fire_enabled: true` 时加载 |
| `models/fire_rk3576_fp16.rknn` | RK3576 开启明火检测时推荐 | RK3576 NPU | 明火模型的 RKNN FP16 版本；NPU 路径使用，缺失或加载失败时回退到 `fire_yolov8n.onnx` |
| `models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx` | 语音指令可选 | CPU | SenseVoice 离线语音识别模型，将麦克风语音转成文本 |
| `models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt` | 语音指令可选 | CPU | SenseVoice 识别词表，必须与 `model.int8.onnx` 位于同一目录 |
| `models/yolov8n.pt`、`models/fire_yolov8n.pt` | 运行不需要 | 训练/导出主机 | PyTorch 原始权重，仅用于训练、继续训练或导出 ONNX |

最小可运行目录如下：

```text
demo-aidesk-vision/
├── models/
│   └── yolov8n.onnx
├── config.json
├── requirements.txt
├── setup_env.sh
└── run_gui.sh
```

完整功能推荐目录如下：

```text
models/
├── yolov8n.onnx
├── yolov8n_rk3576_fp16.rknn
├── fire_yolov8n.onnx
├── fire_rk3576_fp16.rknn
└── sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/
    ├── model.int8.onnx
    └── tokens.txt
```

注意：当前 `config.json` 已设置 `fire_enabled: true`，因此实际启动时还需要对应平台的明火模型。若暂不使用明火检测，将该项改为 `false`。

## 3. 一键部署

在项目根目录执行：

```bash
chmod +x setup_env.sh run_gui.sh
./setup_env.sh
```

脚本会执行以下操作：安装 Debian/Ubuntu 系统包、创建 `.venv`、安装 `requirements.txt`、检查必需模型、运行环境自检及全部单元测试。重复执行不会删除已有虚拟环境或模型。

常用选项：

```bash
# 无 sudo 权限，或系统依赖已安装
./setup_env.sh --skip-system

# 网络位于中国大陆时使用阿里云 PyPI 镜像
./setup_env.sh --china-mirror

# 只复查现有环境，不安装/更新任何依赖
./setup_env.sh --check-only

# 指定 Python 解释器
./setup_env.sh --python /usr/bin/python3.13
```

如果不使用脚本，可手动执行：

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip fonts-noto-cjk alsa-utils \
  libgl1 libglib2.0-0 libxcb-cursor0 libxkbcommon-x11-0 \
  libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xinerama0
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=src QT_QPA_PLATFORM=offscreen \
  .venv/bin/python -m usb_person_detector.entrypoints.env_check
```

## 4. 启动与验证

```bash
./run_gui.sh             # 普通窗口
./run_gui.sh --robot     # 全屏机器人模式
./run_gui.sh --lang zh   # 中文界面
```

无显示器或 CI 环境可做无界面冒烟验证：

```bash
QT_QPA_PLATFORM=offscreen timeout 10s ./run_gui.sh
```

`timeout` 返回 124 表示 GUI 在 10 秒内保持运行并被测试命令按预期结束；若程序自身异常退出，会返回其他状态并输出错误。

重新检查环境及测试：

```bash
./setup_env.sh --check-only
```

## 5. RK3576 开发板部署要点

1. 将项目代码和 `models/` 中的 ONNX、RKNN 文件复制到开发板同一项目目录。
2. 确认 NPU 运行库存在：`ls -l /usr/lib/librknnrt.so`。
3. `config.json` 保持 `backend: "auto"`、`camera: "auto"`。程序会优先选 RKNN；NPU 不可用时自动回退到 ONNX。
4. 执行 `./setup_env.sh --china-mirror`，然后运行 `./run_gui.sh --robot`。
5. 从 SSH 启动时，板子的图形桌面必须已有用户登录；`run_gui.sh` 会自动寻找 Wayland/X11 会话。

RKNN 模型必须以目标芯片 RK3576 为目标生成，并与板载 `librknnrt.so` 兼容。转换环境与命令见 `tools/requirements-rknn.txt` 和 `tools/convert_rknn.py`；转换工作应在 x86 主机的独立 `.venv-rknn` 中完成，不要把 `rknn-toolkit2` 安装进运行环境 `.venv`。

## 6. 常见问题

- **中文显示方块**：安装 `fonts-noto-cjk`，退出并重新登录桌面后再启动。
- **找不到摄像头**：运行 `v4l2-ctl --list-devices` 或检查 `/dev/video*`；RK3576 上建议保持 `camera: "auto"`。
- **Qt 无法加载 xcb**：确认本文列出的 `libxcb-*`、`libxkbcommon-x11-0` 已安装；Wayland 桌面优先直接运行 `run_gui.sh`。
- **NPU 未启用**：确认架构是 aarch64、`/usr/lib/librknnrt.so` 存在、RKNN 文件路径正确；再运行环境自检查看提示。
- **语音按钮不可用**：确认 SenseVoice 目录下同时存在 `model.int8.onnx` 和 `tokens.txt`，并确认 `arecord -L` 能列出麦克风。
- **明火模型加载失败**：确认 `fire_enabled` 与模型是否准备好；不需要该功能时设为 `false`。
- **PySide6 闪退**：项目固定使用 `PySide6-Essentials==6.8.3`，不要自行升级到已知有问题的版本。
