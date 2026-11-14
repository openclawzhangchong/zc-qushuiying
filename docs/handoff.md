# 交接文档（Handoff）

给接手这个项目的人（或未来的自己）：**当前状态、怎么继续、已知坑**。

最后更新：2026-11-14，对应版本 `v1.3.0`（真实竖屏样本实测），提交见 git log。

---

## 1. 这个项目是什么

把开源项目 [wiltodelta/remove-ai-watermarks](https://github.com/wiltodelta/remove-ai-watermarks)
包装成「局域网内浏览器就能用」的中文 WebUI，并补上上游刻意不支持的
**即梦 / Dreamina 视频水印**去除能力。

**核心设计决策：上游代码不入库。**
本仓库只含部署层（WebUI、脚本、文档），部署时用 `deploy.ps1` 克隆上游到本地。
好处是上游更新可以直接 `git pull`；代价是部署步骤必须严格按脚本走。

---

## 2. 当前状态

### 2.1 已完成

| 功能 | 状态 | 验证方式 |
| --- | --- | --- |
| 中文 WebUI（上传/队列/预览/下载） | ✅ 可用 | 浏览器实测 |
| 局域网访问 | ✅ 可用 | `192.168.168.200:8000` 返回 200 |
| 8 类已注册水印（Sora/Veo/Seedance 等） | ✅ 可用 | 上游能力，未逐个实测 |
| **即梦四角轮换水印** | ✅ 可用 | 笔画能量降幅 94%~96% |
| **@昵称长度自适应** | ✅ 可用 | 合成图 2→12 字线性增长；含简单字（十/三/之）的长昵称已修 |
| 自动剪尾 5 秒 | ✅ 可用 | 12s→7.08s；3s 视频跳过 |
| GPU 加速（V100） | ✅ 可用 | 671s → 38s（MI-GAN） |
| GPU 失败自动回退 CPU | ✅ 已实测 | ORT 1.22.0 故障注入：首跑 2.1s 抛错 → CPU 重跑 98 帧完成 |
| **竖屏 9:16** | ✅ 已实测 | 真实竖屏样本 + 合成素材端到端通过（v1.3.0，能量降 94~96%） |
| **亮背景** | ✅ 已实测 | 白占比 >30% 守卫 → 干净走四角兜底 |
| 预览不闪 / 可完整播放 | ✅ 已修 | `domReplaced: false`，可跳到结尾 |
| 部署脚本（PS 5.1） | ✅ 已修 | 修 stderr 陷阱 + uv 阿里云镜像 + 补装 fastapi，干净克隆全链路跑通（含端到端任务） |

### 2.2 未验证 / 有风险

| 项 | 风险 | 说明 |
| --- | --- | --- |
| **其他竖屏分辨率/版式** | 低 | 真实 720×1280 样本已闭环；竖屏水印保持绝对像素尺寸（实测），药丸边距与横屏不同（已并入实测框）。若即梦改版需重测 |
| **即梦改版** | 中 | 若 logo 版式大改（横排、无昵称、logo 位置变化），判据可能失效 |
| **`PILL_REL` 药丸位置** | 低 | 16:9 写死常量；竖屏并入真实样本实测框 (18,14,73,32)@720×1280，即梦若调整需重标 |
| **全新机器部署** | 低 | 干净克隆全链路已跑通（PS 5.1）；uv 引导、冷缓存、无管理员防火墙三条路径未覆盖 |
| 时间一致性（光流） | 低 | 功能可用但默认关闭（慢 28 倍），质量未做对照验证 |

### 2.3 当前运行实例

| 项 | 值 |
| --- | --- |
| 部署目录 | `D:\remove-ai-watermarks` |
| 仓库工作副本 | `D:\zc-qushuiying` |
| 端口 | 8000 |
| 局域网地址 | `http://192.168.168.200:8000/` |
| 公网地址 | `https://shuiyin.jiuyeke.cn/`（**需口令**；局域网/本机免口令） |
| 穿透链路 | Cloudflare(DNS only) → 服务器 Caddy(443) → frps:8081 → 本地 frpc → 8000 |
| 开机自启 | 计划任务 `webui-shuiyin`（BootTrigger+SYSTEM，幂等） |
| 桌面快捷方式 | `去水印WebUI.lnk`（双击=确保服务在跑+打开浏览器） |
| 创作者中心 | `https://shuiyin.jiuyeke.cn/ai/index.html`（五列作品墙 + 虚拟积分 + 模态框；净化接口 `/api/ai/videos` 不含去水印元数据；同受口令保护） |
| 无背景音乐变体 | 每个输出自动生成 `-无背景音乐` 后缀版本（ffmpeg -an 流复制）；模态框可切换播放/分别下载 |
| 口令配置 | `webui/gate_passcode.txt` 或环境变量 `WEBUI_PASSCODE`（不入库） |
| 监听 | `0.0.0.0:8000` |
| 防火墙规则 | `Remove AI Watermarks WebUI`（TCP 8000） |
| GPU | Tesla V100-PCIE-32GB，`CUDAExecutionProvider` 生效 |

> 这些值随机器而变。以 `curl http://127.0.0.1:8000/api/info` 的实际返回为准。

---

## 3. 代码地图

```
zc-qushuiying/
├── deploy.ps1                    # 一键部署（克隆上游、装 WebUI、装依赖、GPU、防火墙）
├── start.bat                     # 启动（需从仓库根目录运行）
├── README.md                     # 面向使用者的说明
├── CHANGELOG.md                  # 版本变更记录
├── docs/
│   ├── screenshot.jpg
│   ├── handoff.md                # ← 本文
│   ├── jimeng-watermark-notes.md # 即梦算法完整技术记录（重要）
│   └── gpu-and-ops-notes.md      # GPU 版本矩阵 + 运维速查
├── tests/
│   └── test_jimeng_detect.py    # 检测器单元测试（14 例，零依赖可跑）
└── webui/
    ├── app.py                    # FastAPI：上传/队列/Range 预览/下载/自动识别
    ├── erase_engine.py           # 区域擦除引擎（复用上游填充+编码管线）
    ├── jimeng_detect.py          # 即梦自动定位（四角轮换 + 昵称自适应）
    ├── gpu_accel.py              # GPU 优先 + 自动回退 CPU
    └── static/index.html         # 中文单页界面（无 CDN 依赖）
```

### 3.1 各文件职责边界

| 文件 | 负责 | 不负责 |
| --- | --- | --- |
| `app.py` | HTTP 接口、任务队列、文件生命周期、调用上游 CLI / 擦除引擎 | 不实现填充算法 |
| `erase_engine.py` | 逐帧应用区域掩码、调用填充后端、编码输出 | 不决定「区域在哪」 |
| `jimeng_detect.py` | 决定「每个时段擦哪里」 | 不执行擦除 |
| `gpu_accel.py` | 推理设备选择与回退 | 不改业务逻辑 |

**关键边界**：`jimeng_detect` 只产出
`[(start_frame, end_frame, (x,y,w,h)), ...]`，`erase_engine` 消费它。
两者可以独立测试、独立替换。

---

## 4. 数据流

```
浏览器上传
   ↓ POST /api/upload（mode / backend / preset / trim_end / temporal）
Job 入队（串行，一次一个）
   ↓ worker_loop
run_job(job)
   ├─ mode == "erase"  → erase 分支
   │    ├─ preset == "jimeng"
   │    │    ├─ jimeng_detect.detect_timeline()   ← 逐帧定位
   │    │    └─ regions_by_frame()                ← 展开成每帧区域
   │    └─ erase_engine.erase_video_regions(regions=逐帧列表)
   │         └─ 每帧：合并掩码 → fill() → 编码写入
   │              失败 → gpu_accel.reset() + CPU 重跑
   ├─ mode == "all"      → 上游 CLI `video all`
   ├─ mode == "metadata" → 上游 CLI `video metadata`
   └─ mode == "auto"     → detect_jimeng() 判断后走上面任一条
   ↓
可选剪尾（ffmpeg -c copy）
   ↓
输出 webui/data/outputs/<job_id>/clean.mp4
   ↓ GET /media/<job_id>/<name>（Range）
浏览器预览 / GET /download/<job_id>
```

---

## 5. 怎么验证改动没坏

### 5.1 最小冒烟测试

```powershell
# 1) 服务活着
curl.exe -s http://127.0.0.1:8000/api/info

# 2) 语法检查
cd D:\remove-ai-watermarks
.venv\Scripts\python.exe -m py_compile webui\app.py webui\erase_engine.py webui\jimeng_detect.py webui\gpu_accel.py

# 3) 即梦定位单跑（不擦除，只看检测结果）
.venv\Scripts\python.exe webui\jimeng_detect.py "C:\path\to\jimeng.mp4"
```

期望输出：每 4 秒一个时段，三个角各出现一次，结尾黑场列出三个角兜底。

### 5.2 端到端（即梦）

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/api/upload ^
  -F "file=@C:\path\to\jimeng.mp4" ^
  -F "mode=erase" -F "backend=cv2" -F "preset=jimeng" ^
  -F "trim_end=0" -F "temporal=0"
```

然后轮询 `/api/job/<id>`，检查 `log` 里是否有：

```
[detect] 帧 0-95 区域 (1064, 551, 138, 75)
[detect] 帧 96-191 区域 (55, 614, 101, 89)
[detect] 帧 192-287 区域 (1081, 61, 132, 77)
```

### 5.3 效果验证（重要）

不要靠肉眼看截图。用**笔画能量**对比，且必须有对照组：

- 水印区域：能量应下降 90%+（实测 94%~96%）
- 无水印的角：变化应在 ±3% 内（实测 ±2.7%）
- 全片改动像素占比：应在 1% 量级（实测 1.01%）

具体算法见 `docs/jimeng-watermark-notes.md` 第 8 节。

---

## 6. 发布流程

### 6.1 本地提交

```powershell
cd D:\zc-qushuiying
git add -A
git commit -m "说明"
```

**中文提交信息不要用 PowerShell 内联传参**（会被 GBK 破坏）。
用 Python 脚本调用 `subprocess` 并指定 `encoding="utf-8"`：

```python
import subprocess
msg = "你的提交信息..."
subprocess.run(["git", "commit", "-m", msg], cwd=r"D:\zc-qushuiying",
               capture_output=True, encoding="utf-8")
```

### 6.2 推送

先试直连：

```powershell
git push origin main
```

失败（`Empty reply from server` / `Failed to connect`）则走 API。
API 推送脚本逻辑见 `docs/gpu-and-ops-notes.md` 第 8.3 节，步骤是：

1. 读远端 `refs/heads/main` 拿当前 SHA
2. 逐个文件 `POST /git/blobs`（**用 base64 编码**，避免行尾/编码问题）
3. `POST /git/trees` 建树（`base_tree` = 远端当前树）
4. `POST /git/commits`（`parents` = 远端 SHA）
5. `PATCH /git/refs/heads/main` 移动分支

**要点**：读取本地文件内容时必须用 `git cat-file blob`（保留原始字节），
不要用 `git show`（会做行尾转换，导致远端内容与本地不一致）。

### 6.3 推送后必查

```powershell
gh api repos/<owner>/zc-qushuiying/commits/main --jq ".sha[0:8] + \" \" + .commit.message[0:40]"
```

并抽查远端文件内容没有被编码破坏（中文可读、关键代码存在）。

---

## 7. 已知坑（务必先读）

按「踩中概率」排序。

### 7.1 中文文件禁用 PowerShell 批量替换

`Get-Content` 按系统默认编码（GBK）解码 UTF-8 文件，会**整file 变成乱码**，
且可能合并行。曾因此毁掉 `app.py`，从 git 恢复后重做。

**一律用 Python：**

```python
from pathlib import Path
p = Path("webui/app.py")
s = p.read_text(encoding="utf-8")
s = s.replace(old, new)
p.write_text(s, encoding="utf-8")
```

### 7.2 `uv sync` 会覆盖上游源码

不要直接改 `remove-ai-watermarks/src/**`。任何对上游行为的修改都应为
**进程内猴子补丁**（参考 `gpu_accel.py` 替换 `_get_session`）。

### 7.3 检测器只在真正 `run()` 时才报 GPU 错误

`get_available_providers()` 里出现 `CUDAExecutionProvider` **不代表能用**。
V100 上的 `cudaErrorNoKernelImageForDevice` 只在执行算子时抛出。
验证必须真正跑一次推理。

### 7.4 时间中位数是即梦方案的基础

任何针对即梦的调试都应先看「时间中位数图」。单帧分析会被运动背景严重干扰，
导致误判（我曾据此怀疑视频有旋转、怀疑箭头不存在）。

### 7.5 窗口尺寸会影响长昵称

`jimeng_detect.WINDOW_W_FRAC = 0.55`。改成更小的值会截断长昵称；
改大则可能让背景物体进入搜索窗造成误报。改动此值必须重跑昵称长度测试。

### 7.6 药丸要用「区域结构复杂度」验证

「AI生成」药丸是半透明白底 + 深色字，**白色判据抓不到**。
用区域内亮度标准差（原始 34 → 处理后 7~19）。

### 7.7 `deploy.ps1` 必须带 UTF-8 BOM

PowerShell **5.1**（Windows 10/Server 默认版本）在没有 BOM 时按系统 ANSI
（中文环境即 GBK）解码脚本文件。脚本里的中文注释/字符串会把字节序列
喂给解析器，某些组合被误判为引号或括号，于是报出**完全误导**的语法错误：

```
表达式或语句中包含意外的标记 '}'。
表达式或语句中包含意外的标记 '"nvidia-cudnn-cu12==9.10.2.21"'。
```

代码本身没问题——是编码问题。修法是给 `.ps1` 加 BOM：

```python
from pathlib import Path
p = Path("deploy.ps1")
text = p.read_bytes().decode("utf-8").lstrip("\ufeff")
p.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
```

**验证方法**（改完必跑）：

```powershell
$err = $null
[System.Management.Automation.Language.Parser]::ParseFile(
    (Resolve-Path deploy.ps1).Path, [ref]$null, [ref]$err) | Out-Null
if ($err.Count) { $err | ForEach-Object { $_.Message } } else { "OK" }
```

### 7.8 `.bat` 必须 CRLF + 纯 ASCII

两个独立问题，都会让批处理「看起来被合并成一行」：

1. **行尾必须是 CRLF**。LF-only 的 `.bat` 在某些 cmd.exe 调用路径下
   会把多行粘连，`rem` 注释可能吞掉后面的真实命令。
2. **含中文必须谨慎**。cmd.exe 默认代码页是 936（GBK），
   而文件是 UTF-8，中文会显示为乱码；更糟的是乱码字节可能破坏
   `if (` / `)` 的配对判断。

本项目 `start.bat` 因此**只用纯 ASCII**（注释也用英文）。
如确需中文提示，应在文件开头加 `chcp 65001 >nul`，
但仍建议关键控制流不依赖中文输出。

### 7.9 `start.bat` 的两种布局（都是坑源）

项目里曾经同时存在三个「启动脚本」，容易混：

| 路径 | 来源 | 说明 |
| --- | --- | --- |
| `zc-qushuiying/start.bat` | 本项目 | 修正版，支持两种布局 |
| `remove-ai-watermarks/webui/start.bat` | **上游自带** | 只有一行 `..\.venv\Scripts\python.exe app.py`，不检查部署状态、不支持 PORT |

**布局 A（仓库根）**：`start.bat` 与 `remove-ai-watermarks/` 同级
```
zc-qushuiying/
├── start.bat
└── remove-ai-watermarks/{webui,.venv}
```

**布局 B（上游根）**：`deploy.ps1` 把 `start.bat` 装到这里
```
remove-ai-watermarks/
├── start.bat
├── webui/app.py
└── .venv/Scripts/python.exe
```

脚本用 `if exist` 自动识别，先试 A 再试 B。

**踩过的坑**：早期版本只认布局 A，装到布局 B 后去找
`remove-ai-watermarks\webui\app.py`，实际变成
`D:\remove-ai-watermarks\remove-ai-watermarks\webui\app.py`，多一层，
报「Not deployed」。

**另一个坑**：无条件 `pause` 会让后台/脚本调用挂起。
现在只在「出错」且「未设 `ZC_NO_PAUSE`」时 pause。

脚本调用推荐：
```powershell
$env:ZC_NO_PAUSE = "1"
cmd /c "D:\remove-ai-watermarks\start.bat"
```

### 7.10 逐帧区域表可能比视频帧数短

`detect_timeline` 按 1 秒分块，**末尾不足一块的帧会被跳过**。
所以 ``len(regions) < frame_count`` 是常态，不是异常。

后果：擦除循环若直接 `regions[i]` 取下标，最后几帧必崩
（实测 98 帧视频崩在第 96 帧，报 `list index out of range`）。

正确做法：
```python
# app.py：按真实解码帧数生成
frame_count = erase_engine.probe_frame_count(job.input_path)
regions = jimeng_detect.regions_by_frame(tl, w, h, frame_count=frame_count)

# erase_engine：先判边界再取下标，超范围沿用最后一个区域
if frame_index < len(regions):
    frame_regions = regions[frame_index]
else:
    frame_regions = regions[-1] if regions else []
```

**注意不要用 `CAP_PROP_FRAME_COUNT`**：它按容器元数据估算。
实测 `-c copy` 剪出的 4s 片段元数据写 96 帧、实际解码 98 帧，
差这 2 帧就足以触发崩溃。必须用 `probe_frame_count()` 逐帧读。

### 7.11 `CUDAExecutionProvider` 存在 ≠ GPU 可用

这是本项目最隐蔽的坑。

```python
"CUDAExecutionProvider" in ort.get_available_providers()   # ← 不够
```

只说明 wheel 编译时带了 CUDA 支持。能否在你的显卡上执行是另一回事：

| 检查层级 | 能发现什么 |
| --- | --- |
| `get_available_providers()` | wheel 是否编译了 CUDA |
| `session.get_providers()` | provider 是否成功加载（DLL 齐全） |
| **真正 `session.run()`** | **显卡 kernel 是否兼容** |

V100 配 onnxruntime-gpu 1.20+ 的典型表现：前两层都通过，
第三层抛 `cudaErrorNoKernelImageForDevice`。

所以 `_cuda_usable()` 必须做一次真实推理自检（见
`gpu_accel._cuda_selftest()`，用 MI-GAN 模型跑 32×32 空图）。
否则界面会长时间谎报「GPU 加速」，任务跑几十帧后才崩、再整段重跑。

### 7.12 验证指标要匹配后端类型

用「文字笔画能量」（拉普拉斯）验证擦除效果时，**MI-GAN 会被冤枉**：

| 后端 | 能量/背景比 | 白像素清除率 | 实际质量 |
| --- | --- | --- | --- |
| MI-GAN | 0.61x | 94.5~97.4% | 纹理自然 |
| cv2 | 0.09x | 100% | 过度平滑，可能发糊 |

同一段水印，MI-GAN 的「降幅」只有 70%，cv2 有 95%+。
曾据此误判 MI-GAN 有残留，实际白像素已清 94%+、检测器也不再命中。

**正确判据**：
1. 白像素是否归零（硬指标）
2. 填充区能量是否显著高于周围背景（残留会高数倍；0.5~1.6x 属正常纹理）

不要用「锐利度」评价生成式模型——它天生会带来新纹理。

### 7.13 别用错启动脚本

调试时我曾用 `D:\remove-ai-watermarks\webui\start.bat`（上游自带），
它不是本项目维护的版本，行为不同（无 PORT 覆盖、无部署检查）。
**用上游根或仓库根的 `start.bat`**，不要用 `webui/` 里的那个。


### 7.10 PS 5.1 会把原生命令的 stderr 进度当终止性错误

`$ErrorActionPreference = "Stop"` 时，git/uv/nvidia-smi 写到 stderr 的
普通进度输出（`Cloning into ...`）会让脚本当场退出。deploy.ps1 用
`Invoke-Native`（临时 EAP=Continue + `$LASTEXITCODE`）包裹全部原生命令；
自己写 PS 部署脚本时同样适用。

### 7.11 `uv sync` 装不出 WebUI 依赖

上游 pyproject 不含 fastapi/uvicorn/python-multipart（其 webui 独立），
`uv sync --frozen` 只装锁文件里的东西——全新部署后 `app.py` 直接
`ModuleNotFoundError`。deploy.ps1 在 sync 后显式补装。另外 `uv sync`
默认直连 pypi.org，国内会卡死（实测十几分钟零进展）——deploy.ps1 已设
`UV_DEFAULT_INDEX` 走阿里云镜像；手动 `uv sync` / `uv pip install` 时
记得同样设置。

---

## 8. 下一步建议（按优先级）

v1.2.0 已完成原 P0 验证清单（竖屏 / 1080p / 亮背景 / GPU 故障注入 /
干净克隆部署）和 P2 的单元测试、回退故障注入。剩余事项：

### P0：真实素材最终确认

1. ~~用真实的竖屏即梦视频测一次~~ ✅ v1.3.0 已完成（shuping.mp4，
   720×1280）：发现并修复 4 个竖屏专属问题（左中轮换位、昵称左伸、
   星标顶尖、片尾动画保护），端到端能量降 94~96%、片尾动画零损伤。
2. **收集更多真实样本**：其他竖屏分辨率（1080×1920）、更多昵称形态
   （无「」括号的短昵称、纯英文昵称）。

### P1：补齐能力

3. **手动画框界面**：界面里允许用户拖框指定擦除区域。
   适用于即梦改版、其他未注册厂商、或用户想擦别的东西。
   后端 `erase_engine` 已支持任意区域列表，只差前端与一个 API。
4. **多视频批量上传**：当前是单文件上传 + 串行队列，批量会明显提升效率。

### P2：工程质量

5. **加 `/api/health`**：区分「服务活着」和「模型可加载」。
6. **给合成素材生成器入库**：`tests/` 里目前只有静态用例，
   把「生成视频 → 检测 → ground truth 比对」的端到端链路也固化下来
   （生成脚本在 D:	estwork\make_synthetic.py，尚未入库）。

---

## 9. 关键数字速查

| 项 | 值 |
| --- | --- |
| 即梦 logo 换角周期 | **4 秒** |
| 药丸位置（相对） | `(0.0047, 0.0111, 0.0625, 0.0556)` |
| 箭头左扩系数 | 字高 × **1.9** |
| wordmark 高度比阈值 | **≤ 1.15** |
| 搜索窗 | 55% 宽 × 34% 高 |
| 分块大小 | 1 秒（24 帧 @24fps） |
| 短空档合并阈值 | ≤ 1.2 秒 |
| MI-GAN 单帧（GPU） | 39 ms |
| 光流 stabilize 单帧（CPU） | 1108 ms |
| 即梦 16s 视频（cv2 GPU） | 25.6 秒 |
| 笔画能量降幅 | 94%~96% |
| 亮背景守卫（窗口白占比） | > **30%** 判判据失效（真实水印块 1%~14%） |
| 阈值自适应基准宽 | **1280px**（更窄按比例放宽，有下限） |
| 昵称链跨洞续链上限 | **4.5×字高**（向右；须同行。向左仅正常字距） |
| 竖屏轮换位（实测） | 右下(y 0.86H) → **左中(y 0.48~0.54H)** → 右上(y 0.04H) |
| 竖屏药丸实测 | (18,14,73,32)@720×1280（边距与横屏不同） |
| 长空档兜底 | 片中全程擦已知角；**片尾只擦最后检出后 1s**（保护片尾动画） |
| **V100 GPU 组合** | `onnxruntime-gpu==1.19.2` + `nvidia-cudnn-cu12==9.10.2.21` |

---

## 10. 参考

- 上游项目：https://github.com/wiltodelta/remove-ai-watermarks
- 上游 PARKED 说明：`remove-ai-watermarks/docs/module-internals.md`（搜 jimeng）
- 上游水印注册表：`remove-ai-watermarks/src/remove_ai_watermarks/video.py`
- 本项目仓库：https://github.com/openclawzhangchong/zc-qushuiying
- 算法细节：`docs/jimeng-watermark-notes.md`
- GPU 与运维：`docs/gpu-and-ops-notes.md`
