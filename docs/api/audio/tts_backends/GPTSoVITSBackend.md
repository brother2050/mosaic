# `GPTSoVITSBackend`

**模块**：`mosaic.nodes.audio.tts_backends.implementations.sovits_backend`
**继承**：`TTSBackend`

## 描述

GPT-SoVITS TTS 后端。

将 GPT-SoVITS 的文本前端（:class:`SoVITSTokenizer`）、声学模型
（:class:`GPT2ARModel`）、SoVITS 解码器（:class:`SoVITSDecoder`）与
流式适配器（:class:`StreamAdapter`）组装为统一的 :class:`TTSBackend`，
支持中英日韩粤语阻塞合成与流式合成，并通过参考音频实现零样本语音克隆。

生命周期
--------
1. 构造后端实例（``is_loaded=False``）。
2. 调用 :meth:`load` 加载三层管线（``is_loaded=True``）。
3. 调用 :meth:`synthesize` / :meth:`synthesize_stream` 合成语音。
4. 调用 :meth:`unload` 释放资源。

Examples
--------
>>> backend = GPTSoVITSBackend(model_path="/data/gpt_sovits")
>>> backend.load(device="cuda", dtype="float16")
>>> audio = backend.synthesize("你好，世界", speaker=None, language="zh")

Notes
-----
GPT-SoVITS 模型遵循 **MIT** 许可。输出采样率为 32000Hz。

## 类属性

| 名称 | 值 |
|---|---|
| `name` | `'sovits'` |

## 方法

### `__init__(self, model_path: 'str', gpt_path: 'str | None' = None, sovits_path: 'str | None' = None, ssl_model: 'str' = 'chinese-hubert-base', speaker_encoder_model: 'str' = 'default', language: 'str' = 'zh', streaming_enabled: 'bool' = True, scheduler: 'Any' = None, repo_id: 'str | None' = None) -> 'None'`

初始化 GPT-SoVITS 后端。

Parameters
----------
model_path : str
    GPT-SoVITS 模型根路径。支持两种布局：

    * **HuggingFace 布局**（``lj1995/GPT-SoVITS``）：
      ``s1bert25hz-*.ckpt`` / ``s2G*.pth`` / ``s2D*.pth``
      / ``chinese-hubert-base/`` / ``chinese-roberta-wwm-ext-large/``
      / ``sv/`` / ``hifigan_*`` 等文件和目录。
    * **旧布局**：``gpt/`` 和 ``sovits/`` 子目录。

    若目录不存在或为空且 ``repo_id`` / ``backend_name`` 可用，
    将自动从 HuggingFace 下载模型。
gpt_path : str | None
    GPT 模型单独路径；``None`` 时使用 ``model_path/gpt/``。
sovits_path : str | None
    SoVITS 模型单独路径；``None`` 时使用 ``model_path/sovits/``。
ssl_model : str
    SSL 模型名称或路径，默认 ``"chinese-hubert-base"``。
speaker_encoder_model : str
    说话人编码器名称，默认 ``"default"``。
language : str
    默认语言代码。
streaming_enabled : bool
    是否启用流式合成。
scheduler : Any
    显存调度器实例。
repo_id : str | None
    HuggingFace 仓库 ID（如 ``"lj1995/GPT-SoVITS"``）。
    ``None`` 时使用 ``backend_name="sovits"`` 对应的默认仓库。

### `clone_voice(self, audio: 'AudioData | str', text: 'str', language: 'str' = 'zh', **kwargs: 'Any') -> 'AudioData'`

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

### `clone_voice_stream(self, audio: 'AudioData | str', text: 'str', language: 'str' = 'zh', **kwargs: 'Any') -> 'Iterator[AudioData]'`

流式语音克隆。

Parameters
----------
audio : AudioData | str
    参考音频。
text : str
    目标文本。
language : str
    语言代码。
**kwargs : Any
    额外参数。

Yields
------
AudioData
    逐块音频数据。

### `describe(self) -> 'TTSBackendSpec'`

返回本后端的规格描述。

### `extract_speaker(self, audio: 'AudioData | str') -> 'dict[str, Any]'`

从音频中提取说话人特征。

提取 ``ref_semantic_tokens``（SSL 编码的语义 token）和
``speaker_embedding``（说话人嵌入向量）。可以预计算并缓存，
避免每次合成重复计算。

Parameters
----------
audio : AudioData | str
    参考音频。可以是 :class:`AudioData` 实例或音频文件路径。

Returns
-------
dict[str, Any]
    包含 ``ref_semantic_tokens`` 和 ``speaker_embedding`` 的字典。

Raises
------
RuntimeError
    SSL 模型未加载。

### `list_speakers(self) -> 'list[str]'`

返回预计算的说话人列表。

GPT-SoVITS 的优势是极少样本克隆，内置音色可能不多。

Returns
-------
list[str]
    已保存的说话人名称列表。

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

### `load_speaker(self, name: 'str') -> 'dict[str, Any]'`

加载已保存的说话人特征。

Parameters
----------
name : str
    说话人名称。

Returns
-------
dict[str, Any]
    包含 ``ref_semantic_tokens`` 和 ``speaker_embedding`` 的字典。

Raises
------
KeyError
    说话人名称不存在。

### `save_speaker(self, name: 'str', audio: 'AudioData | str') -> 'None'`

提取并保存说话人特征到本地。

Parameters
----------
name : str
    说话人名称。
audio : AudioData | str
    参考音频。

### `synthesize(self, text: 'str', speaker: 'str | None' = None, language: 'str' = 'zh', speed: 'float' = 1.0, **kwargs: 'Any') -> 'AudioData'`

阻塞式合成完整语音。

完整流程：

1. 检查 ``is_loaded``。
2. 文本校验。
3. 参考音频处理（如果有 speaker）：
   a. 加载参考音频
   b. SSL_Encoder(ref_audio) → ref_semantic_tokens
   c. SpeakerEncoder(ref_audio) → speaker_embedding
4. SoVITSTokenizer.preprocess(text)
5. phoneme_ids = SoVITSTokenizer.tokenize(text, language)
6. 构造 speaker_info = {"ref_semantic_tokens": ..., "speaker_embedding": ...}
7. semantic_tokens = GPT2ARModel.generate(phoneme_ids, speaker_info, ...)
8. SoVITSDecoder.set_reference(ref_features, speaker_embedding)
9. waveform = SoVITSDecoder.decode(semantic_tokens)
10. 如果 speed != 1.0，做时间拉伸处理
11. 返回 AudioData(waveform, sample_rate=32000)

Parameters
----------
text : str
    待合成文本。
speaker : str | None
    说话人名称或参考音频路径；``None`` 使用默认音色。
language : str
    语言代码（``"zh"`` / ``"en"`` / ``"ja"`` / ``"ko"`` / ``"yue"``）。
speed : float
    语速倍率，``1.0`` 为正常语速。
**kwargs : Any
    额外参数，透传给声学模型（``temperature`` / ``top_p`` /
    ``top_k`` / ``max_new_tokens`` 等）。

Returns
-------
AudioData
    合成结果，``metadata`` 含 ``backend``/``text``/``speaker``
    /``language``/``speed``/``duration``/``sample_rate``/``streaming``。

Raises
------
RuntimeError
    后端未加载。
ValueError
    文本为空。

### `synthesize_stream(self, text: 'str', speaker: 'str | None' = None, language: 'str' = 'zh', speed: 'float' = 1.0, chunk_size: 'int' = 4096, **kwargs: 'Any') -> 'Iterator[AudioData]'`

流式合成语音，逐块 yield :class:`AudioData`。

流程：

1. 参考音频处理同 :meth:`synthesize`。
2. 文本前端处理同 :meth:`synthesize`。
3. 创建 StreamSession。
4. GPT2ARModel.generate_stream(phoneme_ids, speaker_info, stream_batch=16)
   - 每 16 个语义 token yield 一次。
5. SoVITSDecoder.decode_chunk(semantic_chunk) → waveform chunk。
6. StreamSession.push(waveform_chunk)。
7. yield AudioData chunks。

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
    逐块音频数据，``metadata`` 中 ``streaming=True``。

### `unload(self) -> 'None'`

释放模型权重与四层管线资源。
