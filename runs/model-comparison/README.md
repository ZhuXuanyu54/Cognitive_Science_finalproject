# 新版模型对照

用普通话真人朗读和英语真人朗读各一对生成混音。每对使用 `run_cafe.py sweep` 的五个条件，共 10 份混音。两套模型使用相同的混音、参考轨和重叠区间。统一以 16 kHz 保存和评分，Conv-TasNet 在其原生 8 kHz 上推理后将结果重采样至 16 kHz。指标为重叠区内两人的平均 SI-SDR 改善量（SI-SDRi，dB）。数值越大，代表分离结果相对于原始混音的改善越多。

| 样本 | 混音数 | Conv-TasNet | MossFormer2 |
| --- | ---: | ---: | ---: |
| 普通话真人 | 5 | 5.82 | 8.60 |
| 英语真人 | 5 | 13.23 | 12.24 |
| 全部 | 10 | 9.53 | 10.42 |

运行环境是 Apple Silicon、CPU、Python 3.12、PyTorch 2.8.0。逐条件分数见 [`scores.csv`](scores.csv)。两对样本是独立朗读录音，经过加权、错时叠加来模拟两人讲话。每种语言只有一对说话人，无法用这次对照估计不同语言整体的效果，也无法从这些数据得出真实咖啡馆环境的排名。

普通话来源为 [AISHELL-1 / OpenSLR 33](https://www.openslr.org/33/)，分别为说话人 S0764 的 [`BAC009S0764W0121`](https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav) 和 S0916 的 [`BAC009S0916W0489`](https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0916W0489.wav)，都来自 [SenseVoice 官方示例](https://github.com/QwenAudio/SenseVoice#data-prepare)。英语来源为 [LibriSpeech / OpenSLR 12](https://www.openslr.org/12/)，沿用原项目中的 [`1688-142285-0007`](https://download.pytorch.org/torchaudio/tutorial-assets/ctc-decoding/1688-142285-0007.wav)（7.060 秒）和 [`3436-172162-0000`](https://github.com/librosa/data/blob/main/audio/3436-172162-0000.hq.ogg)（取开头 5.300 秒）。原声文件未随代码分发。

[2026 年 SR-CorrNet 论文](https://arxiv.org/abs/2603.29097)报告了该系列在公开英语基准上的结果。该模型在 Apple Silicon CPU 上通过了短片推理和 WAV 导出检查。完整样本耗时较长，这次没有完成同条件对照。本地对照表只列完成全部相同条件的模型。
