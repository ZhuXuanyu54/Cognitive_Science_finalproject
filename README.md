# 鸡尾酒会场景下的单通道重叠语音分离与选择性回放

咖啡馆里，多个人的声音被同一部手机录进一条音轨。本项目可把双人或三人混音分为对应数量的音轨，供听者选择主轨并调低其他轨。

## 答辩版入口

- **直接看结果。** 解压后，用浏览器打开 [语音分离实验结果总览](demo/语音分离_实验结果总览.html)。双人十组、三人六组的评分、重叠段试听音频及同步波形都在这个 HTML 文件里，可以断网打开。
- **重新运行分离。** 先安装下方依赖，再使用 `run_cafe.py recording` 分离自己的单声道混音，或用 `sweep3` 复跑附带的三人样本。模型权重需首次联网下载，已有权重时可以通过 `--weights` 指定本地路径。
- **本项目的改动。** `mossformer.py` 和 `three_speaker_backend.py` 对接发布方的模型实现和预训练权重。`separators.py` 统一采样率转换与输出时间轴。`run_cafe.py`、`evaluate.py` 和 `compare_models.py` 负责构造受控混音、配对评分与同条件比较。`cafe_demo/` 是新跑的音轨回放界面，`demo/` 是答辩使用的内嵌结果页。旧模型入口保留供复现实验记录。

- **汇报材料。** [答辩 PPT](docs/答辩PPT.pptx)、[五分钟逐页讲稿](docs/五分钟逐页答辩讲稿.docx)、[课程大作业报告](docs/课程大作业报告.docx)。
- **课程演示。** [DUET 双观测示例](demo/duet/DUET_现场演示.html)和[复现说明](demo/duet/README.md)。结果页和演示页下载后可用浏览器打开。试听包含从公开语料裁切并处理的录音，[样本来源及许可](samples/three-speaker/来源.md)记录了原始片段。

## 安装

建议使用 Python 3.12。在项目目录安装依赖：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 三人语音分离

三人模式使用作者公开的 8 kHz [MossFormer2 WSJ0-3mix 三人预训练权重](https://huggingface.co/alibabasglab/mossformer2-wsj0mix-3spk)，输入一条单声道混音，输出三条估计音轨。现有 16 kHz 双人模式按下方命令使用。两种模型各用自己的权重，录音入口的输出轨数由 `--model` 决定。

安装依赖后，在本项目目录运行：

```bash
python run_cafe.py recording my-three-person-mix.wav \
  --model mossformer2-3spk --output runs/recording-three
python -m http.server 8080 --bind 127.0.0.1
```

打开 [三人试听页](http://127.0.0.1:8080/cafe_demo/?manifest=../runs/recording-three/manifest.json)。可以在原混音与三条估计音轨之间切换，选定主轨并统一调节另外两条轨的播放电平。录音模式下，轨道按模型输出编号。每条轨对应谁，需要试听确认。客观评分需要独立录制的参考原声。

### 三人受控实验与对照

```bash
python run_cafe.py sweep3 \
  --triple 普通话 samples/three-speaker/cmn-1.wav samples/three-speaker/cmn-2.wav samples/three-speaker/cmn-3.wav \
  --triple 英语 samples/three-speaker/en-1.wav samples/three-speaker/en-2.wav samples/three-speaker/en-3.wav \
  --model mossformer2-3spk --output runs/moss-three
```

把上一条命令的模型换成 `srcorrnet-3spk`、输出目录换成 `runs/srcorr-three`，可运行同条件对照。两次结束后运行 `python compare_models.py runs/moss-three runs/srcorr-three --csv runs/three-comparison.csv`。随项目附带的六个原声及其许可列在[样本来源](samples/three-speaker/来源.md)。也可用自己的 WAV 替换。

每个 `--triple` 接收三个不同说话人的单声道 WAV。每条取开头至多 4 秒，取三条的共同可用长度。三个条件分别是三人共同重叠 50% 且等电平、全段共同重叠且等电平、全段共同重叠且第三人低 6 dB。电平以裁齐后原声的全段 RMS 为准，重叠率以裁齐后的片段长度为分母。评分只取三段录音的共同重叠区，按总 SI-SDR 最大值给估计轨和参考轨做一一配对，记录各个说话人的 SI-SDRi、三轨均值和最低值。

对照使用 [SR-CorrNet 的单通道 2–3 人权重](https://huggingface.co/shinuh/sr-corrnet-ss-1ch-wsj-var-2-3spk)，明确指定输出三轨。比较脚本核对混音文件、三条参考文件和评分窗一致。模型间结果限于本次样本。双人和三人实验的混音人数、评分窗与权重不同，各自用于描述对应任务。

[六个条件的同条件验证](runs/three-speaker-validation/README.md)附有两个模型的逐轨分数。可运行 `python cafe_demo/bundle.py runs/moss-three/manifest.json --output 三人分离试听.html`，把自己的受控实验音轨打包成断网可打开的单文件试听页。

三人 MossFormer2 初次运行会下载约 224 MB 的模型权重。离线可传 `--weights /path/to/directory`，目录包含对应版本的 `encoder.ckpt`、`decoder.ckpt`、`masknet.ckpt`。SR-CorrNet 则使用其发布目录里的 `model.pt` 和 `config.yaml`。三人模型代码来源为 [MossFormer2 官方实现](https://github.com/alibabasglab/MossFormer2/tree/main/MossFormer2_standalone) 与 [SR-CorrNet 官方实现](https://github.com/dmlguq456/SR_CorrNet_SS)。

## 双人分离模型

`run_cafe.py sweep` 和 `run_cafe.py recording` 都接受 `--model`。不传参数时使用 `mossformer2`。模型都只接收一条混音，并输出两条音轨。

| 参数 | 公开预训练模型 | 模型采样率 |
| --- | --- | --- |
| `mossformer2` | ClearVoice [MossFormer2_SS_16K](https://huggingface.co/alibabasglab/MossFormer2_SS_16K)，训练数据包含普通话、英语及带噪声和混响的合成混音 | 16 kHz |
| `srcorrnet` | 2026 年发布的 [SR-CorrNet-L + DM](https://huggingface.co/shinuh/sr-corrnet-ss-1ch-wsj-fix-2spk-l-dm)，使用 WSJ0 双人英语权重 | 8 kHz |
| `convtasnet` | 原项目的 TorchAudio Conv-TasNet，使用 Libri2Mix 权重 | 8 kHz |

首次使用会下载公开权重。MossFormer2 的文件约 670 MB，SR-CorrNet-L 的文件约 145 MB。设备默认 CPU；SR-CorrNet-L 在 CPU 上耗时较长，有兼容的 NVIDIA 环境可用 `--device cuda`。短音频按整段推理，保持两条输出轨在同一片段中的顺序。对照实验统一使用 16 kHz 混音、参考轨及评分时间轴，模型按各自原生采样率推理，再把输出重采样回 16 kHz。

已下载的权重可用 `--weights` 指定。MossFormer2 传入官方 `last_best_checkpoint.pt`（[ModelScope 下载](https://modelscope.cn/models/iic/ClearerVoice-Studio/resolve/master/MossFormer2_SS_16K/last_best_checkpoint.pt)），SR-CorrNet-L 传入同时含官方 `model.pt` 与 `config.yaml` 的目录，Conv-TasNet 传入 `conv_tasnet_base_libri2mix.pt`。SR-CorrNet 的 MIT 源代码固定在官方提交 [`7340365`](https://github.com/dmlguq456/SR_CorrNet_SS/tree/7340365b9cc9a021bf7d400f52fce4b88593b67a)，随代码提供。

新版用普通话和英语的真人朗读各一对生成 10 份受控混音。按重叠区平均 SI-SDRi，MossFormer2 为 10.42 dB，原 Conv-TasNet 为 9.53 dB。普通话组 MossFormer2 更高，英语组 Conv-TasNet 更高。默认值按这 10 份混音的平均成绩选择。见[对照说明](runs/model-comparison/README.md)和[逐场景分数](runs/model-comparison/scores.csv)。

## 咖啡馆对照实验

`run_cafe.py sweep` 每组接收两段不同说话人的单声道 WAV（每段至少 2 秒）。第一个文件代表同桌，第二个代表隔壁桌。每组用同一对原声生成五份混音。

| 条件 | 隔壁桌相对于同桌的电平 | 两段音频的重叠窗口 |
| --- | --- | --- |
| 电平对照 | −6、0、+6 dB | 较短原声时长的 50% |
| 重叠对照 | 0 dB | 较短原声时长的 25%、50%、90% |

两类对照共用 0 dB / 50% 这档。电平按各自完整原声的 RMS 设定，评分区间内实际测得的电平差也会单独记录。重叠百分比描述两段录音时间窗的交集，包含发言中的停顿。每个条件都用同一模型推理，在重叠区匹配两路输出与原声并计算 SI-SDR 和 SI-SDRi。比较不同干扰电平时请同时看两项指标，因为混音基线也会随电平变化。

安装依赖后在项目目录运行。示例路径对应[中文、英文咖啡馆样本](samples/cafe/来源.md)，也可换成自己的 WAV。

~~~bash
python run_cafe.py sweep \
  --model mossformer2 \
  --pair 中文 samples/cafe/chinese/source_1.wav samples/cafe/chinese/source_2.wav \
  --pair 英文 samples/cafe/english/source_1.wav samples/cafe/english/source_2.wav \
  --output runs/cafe-demo
python -m http.server 8080 --bind 127.0.0.1
~~~

打开 [本地试听页](http://127.0.0.1:8080/cafe_demo/)。可以跳到标出的重叠区，在同一播放位置切换混音与分离结果，选择想听的音轨，并调节另一路的音量。每个实验条件生成原始混音、两条分离轨、两条参考轨。`scores.csv` 汇总重叠区分数，`manifest.json` 保存参数、配对和文件路径。要复跑，给 `--output` 指定一个新目录，并在试听页 URL 加上 `?manifest=../runs/新目录/manifest.json`。

若要比较模型，保持每次的 `--pair` 参数完全相同，换用不同的 `--model` 和 `--output`。`compare_models.py` 会核对混音文件和评分区间相同，再按平均 SI-SDRi 排序：

~~~bash
python compare_models.py runs/cafe-demo runs/cafe-srcorrnet runs/cafe-convtasnet --csv model-scores.csv
~~~

升级前曾用原 Conv-TasNet 对中文和英文各一组合成对话完成 10 份混音的试跑。保持重叠窗口为 50%，隔壁桌的设定电平从 −6 dB 提高到 +6 dB 时，目标语音的 SI-SDR 如下。

| 样本 | 干扰低 6 dB | 等电平 | 干扰高 6 dB |
| --- | ---: | ---: | ---: |
| 中文 | 20.46 dB | 16.66 dB | 13.55 dB |
| 英文 | 20.88 dB | 17.75 dB | 14.47 dB |

这两组数据用于验证流程和展示对照，样本量不足以比较不同语言或估计真实咖啡馆的效果。

可以用同一界面试听只有一路混音的手机录音。模型会输出两条轨道供手动选择。客观评分需要对齐的干净原声，手机录音用来试听。

~~~bash
python run_cafe.py recording phone-mono.wav --model mossformer2 --output runs/cafe-phone
~~~

打开 [手机录音试听页](http://127.0.0.1:8080/cafe_demo/?manifest=../runs/cafe-phone/manifest.json)。

## 原始双人试跑（Conv-TasNet）

本实验验证：一条由两位说话人叠加而成的单声道混音，能否由预训练语音分离网络拆成两条语音。模型只接收混音；干净原声仅用于合成混音与计算分数。

模型使用 PyTorch / TorchAudio 的官方 `CONVTASNET_BASE_LIBRI2MIX` 权重。Conv-TasNet 用卷积编码器把混音波形转成特征，为两个声源预测掩码，再由解码器还原两条波形。本项目使用已有权重推理，未重新训练。

## 初步结果

两段原声先做均方根电平对齐；第二段在 1.2 秒后开始，并补齐到同一时间轴后相加。评分采用 SI-SDRi，即分离结果相对于原始混音的 SI-SDR 改善量；输出通道会先和原声做最佳一一配对。

| 试跑 | 混音平均 SI-SDR | 分离平均 SI-SDR | 平均 SI-SDRi |
| --- | ---: | ---: | ---: |
| 合成英文双人语音 | 0.02 dB | 18.43 dB | +18.41 dB |
| LibriSpeech 真人英文双人语音 | 0.00 dB | 13.81 dB | +13.81 dB |

真人组两路 SI-SDRi 分别为 13.21 dB 和 14.40 dB。两组试跑的汇总数据保存在 `runs/*/results.json`。

## 复现

复现旧版本的双人试跑时，`run_pilot.py` 使用 Conv-TasNet 和原来的 8 kHz 评分时间轴。在首次运行时，TorchAudio 会下载官方权重。

~~~bash
python run_pilot.py path/to/source_1.wav path/to/source_2.wav --output runs/reproduced
~~~

输入需要是两段非静音单声道 WAV。程序会重采样至 8 kHz 并导出：`mixture.wav`、两条对齐后的干净原声、两条按最佳匹配排序的分离音轨，以及 `results.json`。

训练权重：[TorchAudio Conv-TasNet Libri2Mix](https://docs.pytorch.org/audio/stable/generated/torchaudio.pipelines.CONVTASNET_BASE_LIBRI2MIX.html)。SI-SDR 指标参考：[SDR – Half-baked or Well Done?](https://arxiv.org/abs/1811.02508)。

## 音频来源

合成英文组由 macOS 内置的 Samantha 与 Daniel 语音生成。真人英文组来自 [LibriSpeech（OpenSLR 12）](https://www.openslr.org/12/)，许可为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。音频来源与处理方式见 [原始样本说明](samples/README.md)。
