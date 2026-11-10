# 单声道双人语音分离初步实验

本实验验证：一条由两位说话人叠加而成的单声道混音，能否由预训练语音分离网络拆成两条语音。模型只接收混音；干净原声仅用于合成混音与计算分数。

模型使用 PyTorch / TorchAudio 的官方 `CONVTASNET_BASE_LIBRI2MIX` 权重。Conv-TasNet 用卷积编码器把混音波形转成特征，为两个声源预测掩码，再由解码器还原两条波形。本项目使用已有权重推理，未重新训练。

## 初步结果

两段原声先做均方根电平对齐；第二段在 1.2 秒后开始，并补齐到同一时间轴后相加。评分采用 SI-SDRi，即分离结果相对于原始混音的 SI-SDR 改善量；输出通道会先和原声做最佳一一配对。

| 试跑 | 混音平均 SI-SDR | 分离平均 SI-SDR | 平均 SI-SDRi |
| --- | ---: | ---: | ---: |
| 合成英文双人语音 | 0.02 dB | 18.43 dB | +18.41 dB |
| LibriSpeech 真人英文双人语音 | 0.00 dB | 13.81 dB | +13.81 dB |

真人组两路 SI-SDRi 分别为 13.21 dB 和 14.40 dB。完整参数和每路分数保存在 `runs/*/results.json`。

## 复现

使用 Python 3.11。在首次运行时，TorchAudio 会下载官方权重。

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pilot.py path/to/source_1.wav path/to/source_2.wav --output runs/reproduced
~~~

输入需要是两段非静音单声道 WAV。程序会重采样至 8 kHz 并导出：`mixture.wav`、两条对齐后的干净原声、两条按最佳匹配排序的分离音轨，以及 `results.json`。

训练权重：[TorchAudio Conv-TasNet Libri2Mix](https://docs.pytorch.org/audio/stable/generated/torchaudio.pipelines.CONVTASNET_BASE_LIBRI2MIX.html)。SI-SDR 指标参考：[SDR – Half-baked or Well Done?](https://arxiv.org/abs/1811.02508)。

## 音频来源

合成英文组由 macOS 内置的 Samantha 与 Daniel 语音生成。真人英文组来自 [LibriSpeech（OpenSLR 12）](https://www.openslr.org/12/)，许可为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。为避免把语料副本提交到仓库，音频二进制文件未纳入版本控制；`samples/README.md` 记录了可复现的来源与处理方式。

这是双人流程试跑。后续实验可固定原声对并改变重叠程度，再以三路输出模型测试三人混音。
