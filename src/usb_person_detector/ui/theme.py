"""红白蓝高达配色主题（参考 RX-78-2 涂装）。

白：装甲主色 / 蓝：胸甲与主面板 / 红：告警与锁定 / 黄：V 字天线点缀
"""
from pathlib import Path

ASSETS = (Path(__file__).resolve().parents[3] / "assets").as_posix()

# ---- 调色板 ----
NAVY_DEEP = "#0A1430"    # 窗口底色（驾驶舱深蓝）
NAVY = "#111F45"         # 面板底色
NAVY_LINE = "#2B4A8F"    # 面板分隔线
BLUE = "#1E56C8"         # 主蓝
BLUE_LIGHT = "#3D7BFF"
WHITE = "#F4F6FA"        # 装甲白
WHITE_DIM = "#C9D2E3"
GRAY = "#8593B0"
RED = "#E0262F"          # 告警红
RED_DARK = "#A8161E"
YELLOW = "#F5C400"       # V 字天线黄

FONT_TITLE = "Ubuntu Condensed"
FONT_BODY = "Noto Sans CJK SC"
FONT_MONO = "Noto Sans Mono CJK SC"

# ---- 视频叠加层用到的 BGR 颜色（OpenCV） ----
BGR_RED = (47, 38, 224)
BGR_WHITE = (250, 246, 244)
BGR_BLUE = (200, 86, 30)
BGR_YELLOW = (0, 196, 245)
BGR_NAVY = (69, 31, 17)


QSS = f"""
* {{
    font-family: "{FONT_BODY}", "DejaVu Sans";
    font-size: 13px;
    color: {WHITE};
}}
QMainWindow, #Root {{
    background: {NAVY_DEEP};
}}

/* ===== 顶部装甲标题栏 ===== */
#Header {{
    background: {WHITE};
    border: none;
    border-bottom: 4px solid {RED};
}}
#Emblem {{
    background: {BLUE};
    color: {WHITE};
    font-family: "{FONT_TITLE}";
    font-size: 26px;
    font-weight: 700;
    padding: 0 14px;
    border-right: 6px solid {YELLOW};
}}
#MainTitle {{
    color: {NAVY_DEEP};
    font-family: "{FONT_TITLE}";
    font-size: 26px;
    font-weight: 700;
    letter-spacing: 2px;
}}
#SubTitle {{
    color: {BLUE};
    font-family: "{FONT_TITLE}";
    font-size: 13px;
    letter-spacing: 2px;
}}
#HeaderTag {{
    color: {GRAY};
    font-family: "{FONT_MONO}";
    font-size: 11px;
}}
#StripeRed {{ background: {RED}; }}
#StripeBlue {{ background: {BLUE}; }}
#StripeYellow {{ background: {YELLOW}; }}

/* 运行状态徽章：根据 state 属性切换颜色 */
#ModeBadge {{
    font-family: "{FONT_TITLE}";
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 3px;
    padding: 6px 18px;
    min-width: 130px;
    border: 2px solid {NAVY_DEEP};
}}
#ModeBadge[state="standby"] {{ background: {BLUE}; color: {WHITE}; }}
#ModeBadge[state="booting"] {{ background: {YELLOW}; color: {NAVY_DEEP}; }}
#ModeBadge[state="running"] {{ background: {RED}; color: {WHITE}; }}
#ModeBadge[state="error"] {{ background: {NAVY_DEEP}; color: {RED}; border-color: {RED}; }}

/* 语言切换按钮（白色标题栏上） */
#LangBtn {{
    background: transparent;
    color: {BLUE};
    border: 2px solid {BLUE};
    font-size: 14px;
    min-width: 52px;
    min-height: 22px;
    padding: 2px 10px;
}}
#LangBtn:hover {{ background: {BLUE}; color: {WHITE}; }}

/* ===== 通用面板 ===== */
#Panel {{
    background: {NAVY};
    border: 1px solid {NAVY_LINE};
}}
#PanelTitle {{
    background: {WHITE};
    color: {NAVY_DEEP};
    font-family: "{FONT_TITLE}";
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 2px;
    padding: 4px 10px;
    border-left: 6px solid {RED};
}}
#PanelCode {{
    color: {GRAY};
    font-family: "{FONT_MONO}";
    font-size: 11px;
}}

/* ===== 视频区 ===== */
#VideoLabel {{
    background: #000000;
    border: 1px solid {NAVY_LINE};
    color: {GRAY};
    font-family: "{FONT_TITLE}";
    font-size: 26px;
    letter-spacing: 4px;
}}

/* ===== 状态卡片 ===== */
#InfoCard {{
    background: {NAVY_DEEP};
    border: 1px solid {NAVY_LINE};
}}
#CardStripe[accent="blue"] {{ background: {BLUE_LIGHT}; }}
#CardStripe[accent="red"] {{ background: {RED}; }}
#CardStripe[accent="yellow"] {{ background: {YELLOW}; }}
#CardStripe[accent="white"] {{ background: {WHITE}; }}
#CardTitle {{
    color: {WHITE_DIM};
    font-family: "{FONT_TITLE}";
    font-size: 12px;
    letter-spacing: 2px;
}}
#CardValue {{
    color: {WHITE};
    font-family: "{FONT_MONO}";
    font-size: 18px;
    font-weight: 700;
}}
#CardValue[alert="true"] {{ color: {RED}; }}

/* ===== 参数输入 ===== */
#FieldLabel {{
    color: {WHITE_DIM};
    font-family: "{FONT_TITLE}";
    font-size: 12px;
    letter-spacing: 1px;
}}
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: {NAVY_DEEP};
    color: {WHITE};
    border: 1px solid {NAVY_LINE};
    border-bottom: 2px solid {BLUE_LIGHT};
    padding: 3px 6px;
    min-height: 24px;
    font-family: "{FONT_MONO}";
    selection-background-color: {BLUE};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-bottom: 2px solid {YELLOW};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {GRAY};
    border-bottom: 2px solid {NAVY_LINE};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: {BLUE};
    width: 16px;
    border: none;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-position: top right; border-bottom: 1px solid {NAVY_DEEP}; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-position: bottom right; }}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {BLUE_LIGHT}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url({ASSETS}/arrow_up.svg);
    width: 8px; height: 5px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url({ASSETS}/arrow_down.svg);
    width: 8px; height: 5px;
}}

/* ===== 按钮 ===== */
QPushButton {{
    font-family: "{FONT_TITLE}";
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 2px;
    padding: 6px 10px;
    border: none;
    min-height: 26px;
}}
#BtnStart {{ background: {BLUE}; color: {WHITE}; border-bottom: 4px solid {YELLOW}; }}
#BtnStart:hover {{ background: {BLUE_LIGHT}; }}
#BtnStop {{ background: {RED}; color: {WHITE}; border-bottom: 4px solid {RED_DARK}; }}
#BtnStop:hover {{ background: #FF3B44; }}
#BtnCapture, #BtnBrowse {{ background: {WHITE}; color: {NAVY_DEEP}; border-bottom: 4px solid {BLUE}; }}
#BtnCapture:hover, #BtnBrowse:hover {{ background: #FFFFFF; border-bottom-color: {YELLOW}; }}
#BtnBrowse {{ font-size: 13px; min-height: 20px; padding: 3px 10px; border-bottom-width: 2px; }}
QPushButton:disabled, #BtnStart:disabled, #BtnStop:disabled, #BtnCapture:disabled, #BtnBrowse:disabled {{
    background: #1A2850;
    color: #4A5A80;
    border-bottom-color: #1A2850;
}}

/* ===== 日志 ===== */
QListWidget {{
    background: {NAVY_DEEP};
    border: 1px solid {NAVY_LINE};
    font-family: "{FONT_MONO}";
    font-size: 12px;
    outline: none;
}}
QListWidget::item {{
    padding: 4px 6px;
    border-bottom: 1px solid #18264D;
}}
QListWidget::item:selected {{
    background: {BLUE};
    color: {WHITE};
}}
QScrollBar:vertical {{
    background: {NAVY_DEEP};
    width: 8px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BLUE};
    min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

/* ===== 右侧滚动区（小屏时可滚动） ===== */
#SideScroll, #SideContent {{
    background: transparent;
    border: none;
}}

/* ===== 视图切换（表情脸 / 摄像头） ===== */
#ViewBtn {{
    background: {NAVY_DEEP};
    color: {WHITE_DIM};
    font-size: 12px;
    min-height: 18px;
    padding: 2px 12px;
    border: 1px solid {NAVY_LINE};
}}
#ViewBtn:checked {{
    background: {BLUE};
    color: {WHITE};
    border-color: {BLUE_LIGHT};
}}
#ViewBtn:hover {{ border-color: {YELLOW}; }}
#RobotBtn {{
    background: {RED};
    color: {WHITE};
    font-size: 12px;
    min-height: 18px;
    padding: 2px 12px;
}}
#RobotBtn:hover {{ background: #FF3B44; }}
#Stack {{ background: #000000; }}

/* ===== 底部状态栏 ===== */
#Footer {{
    background: {NAVY};
    border-top: 2px solid {BLUE};
}}
#FooterText {{
    color: {GRAY};
    font-family: "{FONT_MONO}";
    font-size: 11px;
}}

QMessageBox {{ background: {NAVY}; }}
QMessageBox QLabel {{ color: {WHITE}; }}
QMessageBox QPushButton {{ background: {BLUE}; color: {WHITE}; min-width: 80px; }}
"""
