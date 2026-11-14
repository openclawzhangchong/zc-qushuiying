# GPU 加速与部署运维笔记

本文记录在本项目部署过程中，关于 **NVIDIA GPU 加速**、**Windows DLL 加载**、
**国内网络** 三方面的实测结论与排错流程。所有版本组合都是本机实测可用/不可用的结果。

环境：Windows Server，Tesla V100-PCIE-32GB（Volta / sm_70 / 32GB），
驱动 573.96，Python 3.12.14。

---

## 1. 一句话结论

**V100 / P100 / GTX 10xx（Volta 及更早）必须锁定：**

```bat
uv pip install --python .venv\Scripts\python.exe ^
  "onnxruntime-gpu==1.19.2" ^
  "nvidia-cudnn-cu12==9.10.2.21" ^
  nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12
```

**Turing(sm_75) 及更新**（RTX 20/30/40/50、A100、H100 等）可以用最新版
`onnxruntime-gpu` + 配套 CUDA/cuDNN，无需锁定。

---

## 2. 为什么 V100 要锁版本

两个约束同时存在，任何一个不满足就报错：

| 组件 | 支持 Volta 的最后版本 | 更新版本的表现 |
| --- | --- | --- |
| `onnxruntime-gpu` | **1.19.2** | 1.20+ 移除了 sm_70 kernel |
| `nvidia-cudnn-cu12` | **9.10.x** | 9.12+ 放弃 Volta |

### 2.1 onnxruntime-gpu 移除 sm_70

`onnxruntime-gpu` 从 **1.20** 起，官方 wheel 不再编译 Volta 的 kernel。
provider 依然能被识别（`get_available_providers()` 里有
`CUDAExecutionProvider`），但一推理就失败：

```
[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running
Cast node. Name:'Cast_699' Status Message: CUDA error
cudaErrorNoKernelImageForDevice:no kernel image is available for execution
on the device
```

**这个错误的迷惑性很强**：provider 加载成功、模型加载成功、只有真正
执行算子时才失败。容易误判为「驱动问题」或「显存问题」。

### 2.2 cuDNN 9.12+ 放弃 Volta

即使 ORT 降回 1.19.2，如果 cuDNN 是最新版，仍会失败：

```
[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running
Conv node. Name:'Conv_36' Status Message: CUDNN failure 5003:
CUDNN_STATUS_EXECUTION_FAILED_CUDART
```

`CUDNN_STATUS_EXECUTION_FAILED_CUDART` 的关键信息是「cuDNN 自己拒绝在你的
GPU 上执行」，而不是「DLL 找不到」。

### 2.3 版本矩阵（实测）

| onnxruntime-gpu | cuDNN | 结果 |
| --- | --- | --- |
| 1.31.1 | 9.28 | ❌ 需要 CUDA 13，报 `cublasLt64_13.dll` 缺失 |
| 1.31.1 | 9.x | ❌ 即使补齐也报 kernel 缺失 |
| 1.22.0 | 9.28 | ❌ `cudaErrorNoKernelImageForDevice` |
| 1.21.1 | 9.28 | ❌ `cudaErrorNoKernelImageForDevice` |
| 1.19.2 | 9.28 | ❌ `CUDNN_STATUS_EXECUTION_FAILED_CUDART` |
| 1.19.2 | 8.9.7.29 | ❌ `Error 126` 加载 provider DLL 失败 |
| **1.19.2** | **9.10.2.21** | ✅ **可用** |

> 注意 1.19.2 搭配 cuDNN **8** 也不行：1.19.2 的 CUDA 12 wheel 链接的是
> cuDNN 9 的符号，会报 `Error 126`。

---

## 3. Windows DLL 搜索路径（最容易漏的一步）

pip 安装的 CUDA/cuDNN 库位于：

```
.venv\Lib\site-packages\nvidia\cublas\bin\
.venv\Lib\site-packages\nvidia\cudnn\bin\
.venv\Lib\site-packages\nvidia\cufft\bin\
...
```

**Windows 的 DLL 搜索路径不包含这些目录**，`onnxruntime` 的 CUDA provider
加载时会报：

```
FAIL : LoadLibrary failed with error 126 "" when trying to load
"...\onnxruntime\capi\onnxruntime_providers_cuda.dll"
```

或带具体依赖名：

```
which depends on "cublasLt64_12.dll" which is missing  (Error 126)
```

### 解决：导入 onnxruntime 之前注入

```python
import os, sys
from pathlib import Path

nv = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
for bin_dir in nv.glob("*/bin"):
    os.add_dll_directory(str(bin_dir))

# 同时注入 PATH，覆盖某些仍走 PATH 查找的依赖
os.environ["PATH"] = os.pathsep.join(
    [str(d) for d in nv.glob("*/bin")] + [os.environ.get("PATH", "")]
)

import onnxruntime  # 必须在上面之后
```

**顺序很重要**：`import onnxruntime` 之后再调用 `add_dll_directory` 无效。

**句柄要存活**：`os.add_dll_directory()` 返回的句柄如果被垃圾回收，
目录会从搜索路径移除。不要写成临时表达式丢弃返回值。

本项目在 `webui/gpu_accel.py` 的 `_prepare_cuda_dll_search_path()` 里实现。

---

## 4. 逐层排查缺失 DLL 的方法

报错一次只提示**一个**缺失 DLL，需要逐个补齐。流程：

1. 跑一个最小推理脚本，触发错误。
2. 从错误信息里提取 `which depends on "xxx.dll" which is missing`。
3. 找对应 pip 包（下表），装上。
4. 回到第 1 步。

| 缺失的 DLL | 对应 pip 包 |
| --- | --- |
| `cublasLt64_12.dll` | `nvidia-cublas-cu12` |
| `cudnn64_9.dll` / `cudnn_*.dll` | `nvidia-cudnn-cu12` |
| `cufft64_11.dll` | `nvidia-cufft-cu12` |
| `cudart64_12.dll` | `nvidia-cuda-runtime-cu12` |
| `cublasLt64_13.dll` | `nvidia-cublas`（CUDA 13 的新包名，体积巨大） |

**实测踩坑顺序**：补齐 `cublas` → 还缺 `cufft` → 补齐后通过。
不要假设装完 cuDNN 就够了。

### 最小验证脚本

```python
import os, sys
from pathlib import Path
nv = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
for d in nv.glob("*/bin"):
    os.add_dll_directory(str(d))

import numpy as np, onnxruntime as ort
from huggingface_hub import hf_hub_download

print("ort", ort.__version__, ort.get_available_providers())
p = hf_hub_download(repo_id="andraniksargsyan/migan", filename="migan.onnx")
s = ort.InferenceSession(p, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("providers:", s.get_providers())      # 必须出现 CUDAExecutionProvider
r = s.run(None, {
    "image": np.zeros((1, 3, 512, 512), np.uint8),
    "mask": np.zeros((1, 1, 512, 512), np.uint8),
})
print("GPU INFERENCE OK", r[0].shape)
```

**必须真正 `run()` 一次**。只检查 `get_providers()` 会漏掉
`cudaErrorNoKernelImageForDevice`——那个错误只在执行算子时出现。

---

## 5. 加速效果实测

即梦 16 秒视频（1280×720，393 帧），MI-GAN 后端：

| 配置 | 耗时 | 相对 CPU |
| --- | --- | --- |
| CPU + MI-GAN | 671 秒 | 1× |
| **GPU + MI-GAN** | **38 秒** | **17.5×** |
| GPU + MI-GAN + 时间一致性 | 354 秒 | 1.9× |
| GPU + cv2 | 25.6 秒 | 26× |

单帧填补（1280×720，两个区域）：

| 后端 | CPU | GPU (V100) | 加速 |
| --- | --- | --- | --- |
| cv2 | 42 ms | 不走 GPU | — |
| MI-GAN | 1090 ms | **39 ms** | **28×** |
| LaMa | 10760 ms | **116 ms** | **93×** |

### 为什么「GPU + 时间一致性」只快 1.9 倍

时间一致性（光流平滑 `stabilize_filled_frame`）**完全在 CPU 上跑**，
每帧 +1108ms，而 GPU 填补只要 39ms。光流成了唯一瓶颈：

| 操作 | 耗时 |
| --- | --- |
| MI-GAN 填补（GPU） | 39 ms |
| 光流 stabilize（CPU） | **1108 ms** |

所以本项目**默认关闭**时间一致性，界面提供可选开关。

---

## 6. GPU 优先 + 自动回退的设计

目标是「有 GPU 就用，没有或出错就静默用 CPU，绝不让任务失败」。

`webui/gpu_accel.py` 的做法：进程内替换上游 `region_eraser._get_session`，
**不改上游源码**（否则 `uv sync` 会覆盖）。

### 回退链

| 情况 | 行为 |
| --- | --- |
| GPU 可用、空闲显存 ≥ 1GB | 用 `CUDAExecutionProvider` |
| 未装 GPU 版 ORT | 静默用 CPU |
| 驱动 / CUDA / cuDNN 缺失 | 静默用 CPU |
| 显存不足（< 1GB 空闲） | 静默用 CPU |
| **推理中途 GPU 报错** | **禁用 CUDA，整段用 CPU 重跑一次** |
| 设 `WEBUI_ONNX_DEVICE=cpu` | 强制 CPU |

「整段重跑」很重要：GPU 可能在处理到第 200 帧时才 OOM 或报 kernel 错误，
此时已写出的输出不完整，必须清空重来。

### 为什么要用猴子补丁

```python
from remove_ai_watermarks import region_eraser
original = region_eraser._get_session
region_eraser._get_session = make_gpu_first_factory(original)
```

上游 `_get_session` 里写死了 `providers=["CPUExecutionProvider"]`。
改上游源码会在 `uv sync --frozen` 时被覆盖。进程内替换则：
- 不改磁盘上的上游代码
- 保持上游其余行为（模型下载、缓存、推理包装）完全不变
- 出问题可以直接不调用 `install()`，退回原始行为

---

## 7. 状态自检接口

`GET /api/info` 返回当前推理设备：

```json
{
  "ips": ["192.168.168.200"],
  "port": 8000,
  "gpu": {
    "patched": true,
    "device": "cuda",
    "reason": "CUDA 可用（空闲显存 7655MB）",
    "free_vram_mb": 7655
  }
}
```

界面底部会显示「推理设备：GPU 加速 (CUDA) · 空闲显存 7655MB」，
或回退原因（如「未安装 GPU 版 onnxruntime」）。

**注意**：`device: cuda` 是在**初始化时**的探测结果。
如果 ORT 版本不匹配导致推理期失败，会被回退机制捕获并在任务日志里体现：

```
[gpu] 加速推理失败，回退 CPU 重试: CUDA error cudaErrorNoKernelImageForDevice...
[gpu] 已回退 CPU 完成
```

---

## 8. 国内网络问题

### 8.1 HuggingFace 直连超时

MI-GAN / LaMa 模型从 `huggingface.co` 下载，国内直连会卡住。
WebUI 启动时：

```python
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
```

必须在 `huggingface_hub` 首次调用**之前**设置。

### 8.2 pip 大包下载慢

cuDNN 8 有 686MB，官方源在国内可能只有几十 KB/s（实测跑到 20 分钟仍未完成）。
改用阿里云镜像：

```bat
uv pip install --index-url https://mirrors.aliyun.com/pypi/simple/ ^
  "nvidia-cudnn-cu12==9.10.2.21"
```

实测阿里云 3.4 MB/s，686MB 约 3.5 分钟。清华源对该包返回 403。

### 8.3 GitHub 克隆/推送

| 场景 | 可用方案 |
| --- | --- |
| 克隆上游 | 直连失败时用 `https://ghfast.top/https://github.com/...` |
| `git push` 失败（Empty reply / 连接超时） | 改用 **GitHub REST API** 逐文件建 blob → tree → commit → 更新 ref |

`ghfast.top` 只读镜像**不支持写**，推送必须走 API 或直连。

`git push` 常见错误：

```
fatal: unable to access '...': Empty reply from server
fatal: unable to access '...': Failed to connect to github.com:443
```

如果只是偶发，先试 `git config --global http.version HTTP/1.1`；持续不通则走 API。

---

## 9. 脚本编码与行尾（Windows 部署必读）

这两个问题与 GPU 无关，但会让部署脚本「明明是对的却报语法错误」。

### 9.1 `.ps1` 需要 UTF-8 **BOM**

| 项目 | 说明 |
| --- | --- |
| 症状 | `deploy.ps1` 报「表达式或语句中包含意外的标记 '}'」等莫名其妙错误 |
| 原因 | PowerShell 5.1 无 BOM 时按 ANSI(GBK) 解码，中文注释的字节被误判为语法符号 |
| 修法 | 文件开头写 `EF BB BF`，再跟 UTF-8 正文 |
| 验证 | 用 `[Parser]::ParseFile()` 检查，见下方命令 |

```powershell
$err = $null
[System.Management.Automation.Language.Parser]::ParseFile(
    (Resolve-Path deploy.ps1).Path, [ref]$null, [ref]$err) | Out-Null
if ($err.Count) { $err | Select-Object -First 5 | ForEach-Object { $_.Message } }
else { "语法 OK" }
```

PowerShell 7+ 默认按 UTF-8 读取，不受影响。但用户机器上通常是 5.1，
所以 **BOM 必须加**。

### 9.2 `.bat` 需要 CRLF，建议纯 ASCII

| 问题 | 后果 |
| --- | --- |
| LF-only 行尾 | 多行被粘连，`rem` 注释吞掉后续命令 |
| UTF-8 中文 + GBK 代码页 | 显示乱码；乱码字节可能破坏 `if (...)` 括号配对 |

本项目 `start.bat` 用纯 ASCII + CRLF。检查方法：

```powershell
$b = [IO.File]::ReadAllBytes("start.bat")
"纯ASCII: " + (-not ($b | Where-Object { $_ -gt 127 }))
"CRLF数: " + (($b -join ',') -split '13,10').Count
```

转换命令（Python）：

```python
from pathlib import Path
p = Path("start.bat")
t = p.read_bytes().decode("ascii").replace("\r\n", "\n")
p.write_bytes(t.replace("\n", "\r\n").encode("ascii"))
```

## 10. 运维速查

### 服务是否在跑

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object LocalAddress, OwningProcess
curl.exe -s http://127.0.0.1:8000/api/info
```

### 重启服务

```powershell
$c = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($c) { Stop-Process -Id $c.OwningProcess -Force; Start-Sleep 2 }
cmd /c "D:\remove-ai-watermarks\webui\start.bat"
```

> 进程 PID 每次重启都会变，不要记 PID，用端口查。

### 局域网访问不通

按顺序查：

1. 服务是否监听 `0.0.0.0`（不是 `127.0.0.1`）：

   ```powershell
   Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object LocalAddress
   ```

2. 防火墙规则是否存在并启用：

   ```powershell
   netsh advfirewall firewall show rule name="Remove AI Watermarks WebUI"
   ```

3. 从本机用局域网 IP 自测（排除防火墙）：

   ```powershell
   curl.exe -s -o NUL -w "%{http_code}`n" http://192.168.168.200:8000/
   ```

4. 本机 IP 是否变了（DHCP）：

   ```powershell
   Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" }
   ```

### 任务卡住

```powershell
curl.exe -s http://127.0.0.1:8000/api/jobs
```

队列是**串行**的（一次只跑一个任务）。如果某个任务卡住，先取消它：

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/api/job/<id>/cancel
```

### 显存被占满

```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv
```

本项目单任务串行，正常不会占满 32GB。若显存异常高，检查是否有残留 Python 进程：

```powershell
Get-Process python | Select-Object Id, CPU, WorkingSet64
```

---

## 11. 性能调优建议

| 场景 | 建议 |
| --- | --- |
| 追求速度 | 后端选 **cv2**（GPU 场景下 0.04 秒/帧，393 帧仅 25 秒） |
| 追求质量 | 后端选 **MI-GAN**（GPU 下 39ms/帧，边界无痕） |
| 复杂纹理背景 | 后端选 **LaMa**（GPU 下 116ms/帧，质量最佳） |
| 不要开时间一致性 | 光流是 CPU 单线程，比填补慢 28 倍；逐帧已足够干净 |
| 多区域 | 会自动合并为一次前向，不必手动拆批 |

---

## 12. 备忘：本次部署的实际产物

| 项 | 值 |
| --- | --- |
| 部署目录 | `D:\remove-ai-watermarks` |
| 虚拟环境 | `D:\remove-ai-watermarks\.venv`（Python 3.12.14） |
| 便携 ffmpeg | `C:\Users\<user>\.cache\remove-ai-watermarks\ffmpeg\...\ffmpeg.exe` |
| 模型缓存 | `C:\Users\<user>\.cache\huggingface\hub` |
| WebUI 数据 | `D:\remove-ai-watermarks\webui\data\{uploads,outputs}` |
| 监听 | `0.0.0.0:8000`，局域网 `http://192.168.168.200:8000/` |
| 防火墙规则 | `Remove AI Watermarks WebUI`（TCP 8000） |

> 部署目录、IP 都会因机器而异，以 `deploy.ps1` 实际输出的路径与启动时
> 控制台打印的 IP 为准。

---

## 13. 公网域名穿透与口令门禁（v1.5.0，2026-11-14）

去水印 WebUI 现已通过 frp + 服务器 Caddy 暴露到公网域名
**https://shuiyin.jiuyeke.cn/**（远程需口令，局域网/本机免口令）。

### 13.1 链路

```
访客 → Cloudflare DNS(shuiyin.jiuyeke.cn, DNS only)
     → 服务器 Caddy(:443, Let's Encrypt 证书自动签发)
     → reverse_proxy 127.0.0.1:8081
     → frps(:8081, allowPorts 8080-8090)
     → frp 隧道 → 本地 frpc → 127.0.0.1:8000（WebUI）
```

服务器侧改动（Ubuntu 24.04 / 47.119.167.86，root）：

- DNS：Cloudflare API 加 A 记录 `shuiyin -> 47.119.167.86`
  （zone `jiuyeke.cn`，**DNS only**，与 musicapi 等既有记录一致；
  橙云代理会让 Caddy 的 ACME 变复杂，不必开）
- Caddy：站点块追加到 **真正生效的配置**
  `/opt/1panel/apps/caddy/caddy/data/conf/Caddyfile`
  （1Panel 容器把该目录挂载为容器内 `/etc/caddy`——主机上的
  `/etc/caddy/Caddyfile` 是一份游离旧副本，改它不生效！）
  然后 `docker exec 1Panel-caddy-IqWe caddy reload --config /etc/caddy/Caddyfile`
- allowPorts：frps 只允许 8080-8090，llama 占 8080，本服务用 8081

本地侧（部署机）：

- `D:\webui-startup\frpc-shuiyin.toml`（独立 frpc 实例，与
  llamacpp 的 frpc 并存，命令行带 `frpc-shuiyin` 便于区分）
- `D:\webui-startup\start-all.ps1` / `run-webui.cmd` / `run-frpc.cmd`：
  幂等启动脚本（按端口/命令行判重）
- 计划任务 `webui-shuiyin`：BootTrigger + SYSTEM，开机自动拉起
- 桌面快捷方式 `去水印WebUI.lnk`：双击 = 确保服务在跑 + 打开浏览器

### 13.2 口令门禁的实现要点

- 门禁在 **WebUI 应用层**（FastAPI 中间件），不在 Caddy/frp 层：
  这样门禁页可以做中文 UI、Cookie 30 天免重复输入、所有路由
  （含视频预览/下载）统一受控
- 「远程」判定：请求带 `X-Forwarded-For` / `X-Real-IP` /
  `CF-Connecting-IP` 任一转发头（Caddy 必加 XFF），**或** Host 头
  不是本机地址。后者是关键兜底——frps 的 8081 绑定公网 IP，
  直连 `http://47.119.167.86:8081` 的请求没有转发头，但 Host 是
  公网地址，同样要过口令（Host 伪造不了它自己的直连语义）
- 局域网/本机直连：无转发头 + Host 为本机 IP/localhost → 直通
- 口令存放：环境变量 `WEBUI_PASSCODE` 或 `webui/gate_passcode.txt`
  （本地文件不入库）；Cookie 值为口令+固定盐的 SHA-256，不回传明文

### 13.3 运维速查

```powershell
# 服务状态
curl.exe -s http://127.0.0.1:8000/api/info
# 隧道状态（本机 frpc 是否在跑）
Get-CimInstance Win32_Process -Filter "name='frpc.exe'" | % CommandLine
# 重启整套（按端口杀，勿按命令行——上游 start.bat 启动的实例是相对路径）
Get-NetTCPConnection -LocalPort 8000 -State Listen | % OwningProcess   # 然后 Stop-Process
Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','D:\webui-startup\start-all.ps1' -WindowStyle Hidden
# 手动触发开机任务（等价于重启后自动执行）
Start-ScheduledTask -TaskName 'webui-shuiyin'
```

### 13.4 踩坑记录

- **改 Caddyfile 要改对文件**：主机 `/etc/caddy/Caddyfile` 是游离旧副本，
  容器实际读的是 1Panel 挂载路径
  `/opt/1panel/apps/caddy/caddy/data/conf/Caddyfile`（教训：先
  `docker inspect <容器> --format '{{json .Mounts}}'` 确认挂载再动手）
- **PowerShell 脚本含中文必须带 UTF-8 BOM**（本项目已知坑 7.7 的重演：
  无 BOM 时 PS 5.1 按 GBK 解码，注册计划任务的脚本直接解析失败）
- **Start-Process 的 `-WindowStyle` 与 `-RedirectStandard*` 参数集互斥**；
  需要隐藏窗口 + 重定向输出时，用隐藏的 cmd 包一层重定向，
  或把命令写进独立 .cmd 文件再拉起（本项目采用后者，还顺便绕开了
  PowerShell 传参引号问题）
- frpc 多实例并存没问题，但**判重要看命令行里的配置文件名**
  （`frpc-shuiyin.toml`），按进程名判重会把 llamacpp 的 frpc 误判成自己
