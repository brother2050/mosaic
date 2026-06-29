# `FishSpeechBackend`

**模块**：`mosaic.nodes.audio.tts_backends.implementations.fish_backend`
**继承**：`TTSBackend`

## 描述

Fish Speech TTS 后端。

将 Fish Speech 的文本前端（:class:`FishTokenizer`）、声学模型
（:class:`FishLlamaARModel`）、复合声码器（VQDecoder + HiFi-GAN）与
流式适配器（:class:`StreamAdapter`）组装为统一的 :class:`TTSBackend`，
支持中英日韩四语言阻塞合成与流式合成，并通过参考音频实现语音克隆。

生命周期
--------
1. 构造后端实例（``is_loaded=False``）。
2. 调用 :meth:`load` 加载四层管线（``is_loaded=True``）。
3. 调用 :meth:`synthesize` / :meth:`synthesize_stream` 合成语音。
4. 调用 :meth:`unload` 释放资源。

Examples
--------
>>> backend = FishSpeechBackend(model_path="/data/fish_speech")
>>> backend.load(device="cuda", dtype="float16")
>>> audio = backend.synthesize("你好，世界", speaker=None, language="zh")

Notes
-----
Fish Speech 模型遵循 **Apache-2.0** 许可。输出采样率为 22050Hz。

## 类属性

| 名称 | 值 |
|---|---|
| `name` | `'fish'` |

## 方法

### `__init__(self, model_path: 'str', hifi_gan_path: 'str | None' = None, audio_encoder_path: 'str | None' = None, codec_type: 'str' = 'dac', language: 'str' = 'zh', use_flash_attention: 'bool' = True, streaming_enabled: 'bool' = True, scheduler: 'Any' = None, repo_id: 'str | None' = None) -> 'None'`

初始化 Fish Speech 后端。

Parameters
----------
model_path : str
    Fish Speech 模型权重路径。支持两种布局：

    * **HuggingFace 布局**（``fishaudio/fish-speech-1.5``）：
      ``config.json`` / ``model.pth`` /
      ``firefly-gan-vq-fsq-8x1024-21hz-generator.pth`` /
      ``tokenizer.tiktoken`` / ``special_tokens.json``。
    * **旧布局**：``vocab.json`` / ``acoustic_model.*`` /
      ``vq_decoder.safetensors`` / ``hifi_gan.safetensors``。

    若目录不存在或为空且 ``repo_id`` / ``backend_name`` 可用，
    将自动从 HuggingFace 下载模型。
hifi_gan_path : str | None
    HiFi-GAN 权重路径；``None`` 时按候选列表查找
    （``firefly-gan-vq-fsq-8x1024-21hz-generator.pth`` 等）。
audio_encoder_path : str | None
    AudioEncoder 权重路径（语音克隆用）；``None`` 时不加载
    AudioEncoder，语音克隆功能不可用。
codec_type : str
    音频编码器类型，``"dac"`` / ``"encodec"`` / ``"snac"``。
language : str
    默认语言代码。
use_flash_attention : bool
    声学模型是否使用 Flash Attention。
streaming_enabled : bool
    是否启用流式合成。
scheduler : Any
    显存调度器实例。
repo_id : str | None
    HuggingFace 仓库 ID（如 ``"fishaudio/fish-speech-1.5"``）。
    ``None`` 时使用 ``backend_name="fish"`` 对应的默认仓库。

### `clone_voice(self, audio: 'Any', text: 'str', language: 'str' = 'zh', **kwargs: 'Any') -> 'AudioData'`

语音克隆的便捷方法。

输入参考音频 + 目标文本，合成与参考音频音色相同、内容为目标文本
的语音。

Parameters
----------
audio : AudioData | str
    参考音频。可以是 :class:`AudioData` 实例或音频文件路径。
text : str
    目标文本。
language : str
    语言代码。
**kwargs : Any
    额外参数。

Returns
-------
AudioData
    克隆语音结果。

### `describe(self) -> 'TTSBackendSpec'`

返回本后端的规格描述。

### `list_speakers(self) -> 'list[str]'`

返回内置的说话人列表。

Fish Speech 主要通过参考音频实现任意音色克隆，内置音色较少。

Returns
-------
list[str]
    内置音色标识列表。

### `load(self, device: 'str' = 'cuda', dtype: 'str' = 'float16') -> 'None'`

加载模型权重并组装四层管线。

通过 :class:`Scheduler` 解析设备与管理显存：显存不足时尝试释放
其他已加载节点；仍不足则抛出 :class:`MemoryError`。

Parameters
----------
device:
    目标设备，默认 ``"cuda"``；无 GPU 时自动降级为 ``"cpu"``。
dtype:
    权重精度，如 ``"float16"`` / ``"float32"`` / ``"bfloat16"``。

Raises
------
MemoryError
    GPU 显存不足以加载本后端。
ImportError
    缺少必要依赖。
RuntimeError
    模型加载失败（附带友好提示）。

### `synthesize(self, text: 'str', speaker: 'str | None' = None, language: 'str' = 'zh', speed: 'float' = 1.0, **kwargs: 'Any') -> 'AudioData'`

阻塞式合成完整语音。

完整流程：

1. 检查 ``is_loaded``。
2. 文本校验。
3. 文本预处理（``FishTokenizer.preprocess``）。
4. 说话人/参考音频编码（``FishTokenizer.encode_speaker``）。
5. 分词（``FishTokenizer.tokenize``）→ token_ids。
6. 合并推理参数。
7. 声学模型生成（``FishLlamaARModel.generate``）→ audio_codec_ids。
8. 复合声码器解码（``_CompositeVocoder.decode``）→ waveform。
9. 构造 :class:`AudioData` 返回。

Parameters
----------
text : str
    待合成文本。
speaker : str | None
    说话人标识。Fish Speech 中可以是参考音频文件路径或预编码的
    codec token ids；``None`` 使用默认音色。
language : str
    语言代码（``"zh"`` / ``"en"`` / ``"ja"`` / ``"ko"``）。
speed : float
    语速倍率。
**kwargs : Any
    额外参数，透传给声学模型（``temperature`` / ``top_p`` /
    ``top_k`` / ``max_new_tokens`` 等）。

Returns
-------
AudioData
    合成结果。

Raises
------
RuntimeError
    后端未加载。
ValueError
    文本为空。

### `synthesize_stream(self, text: 'str', speaker: 'str | None' = None, language: 'str' = 'zh', speed: 'float' = 1.0, chunk_size: 'int' = 4096, **kwargs: 'Any') -> 'Iterator[AudioData]'`

流式合成语音，逐块 yield :class:`AudioData`。

Parameters
----------
text : str
    待合成文本。
speaker : str | None
    说话人标识。
language : str
    语言代码。
speed : float
    语速倍率。
chunk_size : int
    每个音频块的目标采样数。
**kwargs : Any
    额外参数。

Yields
------
AudioData
    逐块音频数据。

### `unload(self) -> 'None'`

释放模型权重与四层管线资源。
