# 测试脚本说明

本目录收留给即梦水印去除功能写下的测试与验证工具。**运行环境**：部署机的
venv（`D:\remove-ai-watermarks\.venv`），已含 numpy / opencv / PIL；
脚本默认从部署目录导入 webui 模块与上游包，在其他机器上运行需先部署。

| 脚本 | 用途 | 运行方式 |
| --- | --- | --- |
| `test_jimeng_detect.py` | 检测器单元测试（16 例，零依赖可跑） | `python tests/test_jimeng_detect.py` 或 pytest |
| `make_synthetic.py` | 生成合成即梦测试视频（可参数化分辨率/缩放假设/昵称长度/背景亮度），输出到 `D:\testwork\syn\` 并写 ground truth | `python tests/make_synthetic.py` |
| `check_syn.py` | 合成视频检测结果 vs ground truth 逐段比对（角一致/框包含/覆盖率/药丸） | `python tests/check_syn.py` |
| `verify_erase.py` | 端到端擦除效果量化：笔画能量降幅 / 对照组 / 白像素残留 / 药丸 std / PSNR / 改动像素占比 | `python tests/verify_erase.py <原始视频> <处理后视频> [区域json]` |
| `analyze_chunks.py` | 逐 1s 块检测体检：背景亮度、白占比、字形粘连、命中/漏检/误检 | `python tests/analyze_chunks.py <视频>` |

## 典型流程

```powershell
# 1) 单跑检测器（最快定位）
D:\remove-ai-watermarks\.venv\Scripts\python.exe webui\jimeng_detect.py "C:\path\to\video.mp4"

# 2) 单元测试
python tests\test_jimeng_detect.py

# 3) 合成素材回归（分辨率/昵称/亮背景/缩放假设）
python tests\make_synthetic.py
python tests\check_syn.py

# 4) 端到端量化（webui 处理完成后，原始文件与输出文件对比）
python tests\verify_erase.py "C:\path\orig.mp4" "D:\remove-ai-watermarks\webui\data\outputs\<job>\zc-xxxx.mp4"
```

## 验收标准（即梦预设，cv2 后端）

- 水印区域笔画能量（拉普拉斯绝对值均值）降幅 **≥ 90%**（实测 94~96%）
- 静态白像素残留 ≈ 0（用「≥95% 帧皆为白」的静态掩码判定，排除内容亮斑）
- 对照角（同段无水印的角）能量变化 **±3% 内**；改动像素占比 1~2%
- 半透明药丸用「区域亮度 std」判定（横屏 ~34→19，夜景竖屏 28→2）
- **必须做对照组**——只证明水印没了不能证明没误伤

详细方法论见 `docs/jimeng-watermark-notes.md` 第 8/14/15 节。
