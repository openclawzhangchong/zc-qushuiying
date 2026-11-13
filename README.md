# zc-qushuiying · AI 视频水印去除（中文部署版）

基于开源项目 [wiltodelta/remove-ai-watermarks](https://github.com/wiltodelta/remove-ai-watermarks) 的开箱即用部署方案，新增了一个中文 WebUI：浏览器上传视频 → 自动去除可见 AI 水印并清除 AI 元数据 → 网页内预览、下载。局域网内任何设备都能访问。

![WebUI 界面](docs/screenshot.jpg)

## 功能

- 拖拽 / 点击上传视频，支持 mp4 / mov / m4v / webm / mkv / avi / flv
- **自动识别来源**：检测到即梦 / Dreamina 导出时，自动切换到区域擦除 + MI-GAN
- **区域擦除模式**：处理上游未注册的可见水印（即梦左上「AI生成」药丸 + 右下 logo/作者名）
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

## 为什么需要区域擦除模式

上游的视频可见水印检测表只注册了 8 个厂商：`sora / veo / seedance / doubao / dola / hailuo / vidu / kling`。**即梦（Jimeng / Dreamina）不在其中**，这是上游的主动决定而非疏漏——`docs/module-internals.md` 记录该水印被 PARKED：

> 即梦 CN 水印在视频上模板匹配分不开正负样本（1/5 正样本 vs 2/25 负样本），左上「AI生成」药丸的误报模式是静态纹理角，时间递归仲裁器无法拦截。

所以对即梦视频跑 `video all` 会得到「未检测到可见水印」，直接输出直通版本——三个填充后端都没有机会工作。

`webui/erase_engine.py` **不修改上游注册表**，而是复用上游同一条逐帧填充与原子编码管线（同样的 cv2 / MI-GAN / LaMa 后端、色彩档案、VFR 时间戳、HDR 拒绝、音频流复制、元数据剥离），接受手动或预设区域直接擦除。即梦预设区域按 1280x720 实测标定，并按帧尺寸等比缩放。

### 后端性能与效果（1280x720，1280 视频实测）

| 后端 | 速度 | 边界干净度 |
| --- | --- | --- |
| cv2 | ~0.04 秒/帧 | 简单背景可用；复杂纹理区会有涂抹拉丝 |
| MI-GAN | ~1.1 秒/帧 | 干净，即梦 logo 区域无痕（推荐） |
| LaMa | ~3-5 秒/帧 | 最佳，需额外下载约 200 MB 模型 |

> 多区域会合并为**一次**前向计算（两个区域从 2.18s/帧降到 1.09s/帧）。

### 国内网络：HuggingFace 镜像

MI-GAN / LaMa 模型从 HuggingFace 下载，国内直连会超时。WebUI 启动时自动设置：

```
HF_ENDPOINT=https://hf-mirror.com
```

也可用系统环境变量永久生效。

## 目录结构

```
zc-qushuiying/
├── deploy.ps1        # 一键部署：克隆上游、装 WebUI、装依赖、放行防火墙
├── start.bat         # 启动 WebUI（自动定位部署目录）
├── webui/
│   ├── app.py        # FastAPI 服务：上传 / 任务队列 / 预览(Range) / 下载 / 自动识别
│   ├── erase_engine.py  # 区域擦除引擎（复用上游填充与编码管线）
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
| `webui/erase_engine.py` | 区域擦除引擎：绕过上游未注册的即梦水印，复用同一填充/编码管线 |
| `deploy.ps1` / `start.bat` | Windows 一键部署与启动，含防火墙放行与克隆镜像回退 |
| 上游代码 | 不入库，部署时从官方仓库克隆，保持 `uv.lock` 锁定版本 |

## 合规与许可

- 上游项目：Apache-2.0（见 `LICENSE`），上游版权 2025-2026 wiltodelta
- 本仓库的部署层代码同样以 Apache-2.0 提供
- 仅用于处理你自己生成或拥有版权的内容；部分司法辖区（含中国）对移除 AI 标识有限制，使用前请自行确认合规
- 上游文档：[supported signals](https://github.com/wiltodelta/remove-ai-watermarks/blob/main/docs/supported-signals.md) · [known limitations](https://github.com/wiltodelta/remove-ai-watermarks/blob/main/docs/known-limitations.md)
