# DUET 双观测语音分离课堂演示

双击 `DUET_现场演示.html`，用 Chrome、Edge 或 Safari 打开。先在「易」档看两个明显分开的峰，依次试听麦克风 1 的混音、分离轨 1 和分离轨 2。然后切换到「中」档，观察两个峰沿着时延轴靠近，试听效果随之变化。「难」档的两个麦克风观测完全相同，算法检测到一个峰，界面只保留原始混音。

三档固定同一对语音和同一条麦克风 1 混音。第二路给两个声源分别施加相对增益和整数点时移。脚本用 32 ms Hann 窗、8 ms 步长计算 STFT，并在 125–1400 Hz 内从两路复系数比值估计幅度和时延。能量加权二维直方图平滑后查找显著局部峰。用各个峰的归一化复残差将时频点指派给一个峰，再将得到的二值掩蔽施加给麦克风 1 的混音，做 iSTFT。原始 DUET 使用对称幅度估计；这里用幅度比的分贝坐标和二维局部峰做教学演示。

| 条件 | 相对幅度（源 1、源 2） | 相对时延（源 1、源 2） | 双轨平均 SI-SDRi |
|---|---:|---:|---:|
| 易 | −2.38 dB、+2.28 dB | −3、+3 采样点 | +9.94 dB |
| 中 | 0 dB、0 dB | −1、+1 采样点 | +9.36 dB |
| 难 | 0 dB、0 dB | 0、0 采样点 | 单峰，无法给出双轨评分 |

得分是两条估计音轨最佳一一配对后，相对原始混音的平均 SI-SDR 改善量。真值仅用于合成观测和演示后评分。可听波形统一了播放电平。这三档是固定语料的教学示例，不能据此归纳任意录音下的分离质量。

## 复现

依赖 Python、NumPy、SciPy 和 Matplotlib。若图中文字缺失，安装可用的中文字体，或更换 `duet.py` 中的中文字体路径。

```sh
python3 duet.py BAC009S0764W0121.wav BAC009S0916W0489.wav media
python3 build_html.py media DUET_现场演示.html
```

- 语料来自北京希尔贝壳科技有限公司提供的 [AISHELL-1 / OpenSLR 33](https://www.openslr.org/33/)，Apache 2.0。使用 [BAC009S0764W0121.wav](https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav) 和 [BAC009S0916W0489.wav](https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0916W0489.wav)，由 [SenseVoice 公开示例](https://github.com/FunAudioLLM/SenseVoice#data-prepare) 提供。两段录音做截齐、去直流、均衡和模拟时移，嵌入的是混音和分离轨。
- 方法引用 [Yılmaz & Rickard (2004), Blind Separation of Speech Mixtures via Time-Frequency Masking](https://doi.org/10.1109/TSP.2004.828896)。
