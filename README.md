# zc-qushuiying · AI 视频水印去除（中文部署版）

基于开源项目 [wiltodelta/remove-ai-watermarks](https://github.com/wiltodelta/remove-ai-watermarks) 的开箱即用部署方案，新增了一个中文 WebUI：浏览器上传视频 → 自动去除可见 AI 水印并清除 AI 元数据 → 网页内预览、下载。局域网内任何设备都能访问。

![WebUI 界面](docs/screenshot.jpg)

## 功能

- 拖拽 / 点击上传视频，支持 mp4 / mov / m4v / webm / mkv / avi / flv
- 自动检测并移除已注册的可见 AI 水印：Sora 2、Veo、Seedance、豆包、Dola、海螺 AI、Kling AI、Vidu 等
- 清除 AI 元数据：C2PA、EXIF/XMP 生成参数、中国 TC260 `AIGC` 标签等（不重编码视频流）
- 填充后端可选：cv2（快速、纯本地）/ MI-GAN / LaMa（质量更好，首次使用联网下载模型）
- 任务列表实时显示进度与日志；完成后页面内直接预览（支持拖动进度条）并一键下载
- 可选：自动剪掉视频最后 5 秒（流复制截断，不重编码，几乎瞬时完成；视频过短时自动跳过）
- 单任务串行队列；上传与输出文件保留 24 小时后自动清理，也可手动删除
- 全程 CPU 运行，无需 GPU、无需系统 ffmpeg（依赖自带便携版 ffmpeg）

## 快速开始

### 一键部署（Windows）

```powershell
git clone https://github.com/<你的账号>/zc-qushuiying.git
cd zc-qushuiying
powershell -ExecutionPolicy Bypass -File deploy.ps1
```

`deploy.ps1` 会自动完成：克隆上游仓库 → 把 `webui/` 装进去 → `uv sync` 安装依赖（含便携 ffmpeg）→ 放行防火墙端口 → 打印访问地址。

### 启动

```bat
start.bat
```

- 本机访问：http://127.0.0.1:8000/
- 局域网访问：http://<本机IP>:8000/（启动时控制台会打印实际 IP；防火墙规则名 `Remove AI Watermarks WebUI`）

### 手动部署

```powershell
git clone --depth 1 https://github.com/wiltodelta/remove-ai-watermarks
Copy-Item -Recurse -Force zc-qushuiying\webui remove-ai-watermarks\webui
cd remove-ai-watermarks
uv sync --frozen --extra video --extra migan
cd webui
..\.venv\Scripts\python.exe app.py
```

> 需要已安装 [uv](https://docs.astral.sh/uv/)（`pip install uv` 或官方安装脚本）。国内网络克隆/下载失败时，可在 clone 地址前加镜像前缀，例如 `https://ghfast.top/https://github.com/...`。

## 目录结构

```
zc-qushuiying/
├── deploy.ps1        # 一键部署：克隆上游、装 WebUI、装依赖、放行防火墙
├── start.bat         # 启动 WebUI（自动定位部署目录）
├── webui/
│   ├── app.py        # FastAPI 服务：上传 / 任务队列 / 预览(Range) / 下载
│   └── static/
│       └── index.html  # 中文单页界面（无外部 CDN 依赖）
└── remove-ai-watermarks/   # 部署时克隆的上游代码（不入库）
```

## 工作原理

WebUI（FastAPI）把上传的视频交给上游 CLI 的 `video all` / `video metadata` 子命令处理：

- `video all`：扫描全部帧寻找"时间上稳定出现"的注册水印 → 生成掩码 → cv2/MI-GAN/LaMa 填补 → 重编码（libx264），音频无损复制，同时清除 AI 元数据；未检出水印时输出去元数据的直通版本
- `video metadata`：仅做格式感知的元数据清除，视频/音频流原样复制
- 可选剪尾：处理完成后用自带 ffmpeg 以 `-c copy` 流复制方式截掉结尾（WebUI 勾选，默认关闭）
- 处理结果经支持 HTTP Range 的接口回放，浏览器可流畅预览与拖动

## 与上游的差异（本仓库全部内容）

| 内容 | 说明 |
| --- | --- |
| `webui/` | 全新中文 WebUI：上传、任务队列、进度日志、Range 预览、下载、自动清理 |
| `deploy.ps1` / `start.bat` | Windows 一键部署与启动，含防火墙放行与克隆镜像回退 |
| 上游代码 | 不入库，部署时从官方仓库克隆，保持 `uv.lock` 锁定版本 |

## 合规与许可

- 上游项目：Apache-2.0（见 `LICENSE`），上游版权 2025-2026 wiltodelta
- 本仓库的部署层代码同样以 Apache-2.0 提供
- 仅用于处理你自己生成或拥有版权的内容；部分司法辖区（含中国）对移除 AI 标识有限制，使用前请自行确认合规
- 上游文档：[supported signals](https://github.com/wiltodelta/remove-ai-watermarks/blob/main/docs/supported-signals.md) · [known limitations](https://github.com/wiltodelta/remove-ai-watermarks/blob/main/docs/known-limitations.md)
