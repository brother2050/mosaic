# `CosyVoiceBackend`

**模块**：`mosaic.nodes.audio.tts_backends.implementations.cosyvoice_backend`
**继承**：`TTSBackend`

## 描述

CosyVoice TTS 后端。

将 CosyVoice 的文本前端（:class:`CosyVoiceTokenizer`）、Flow Matching
声学模型（:class:`FlowMatchingModel`）、HiFi-GAN 声码器
（:class:`HiFiGanVocoder`）、语音 Tokenizer（:class:`SpeechTokenizer`）
和说话人编码器（:class:`SpeakerEncoder`）组装为统一的
:class:`TTSBackend`。

与自回归后端的关键区别：
* ``acoustic_type="flow_matching"``（非 ``"ar"``）
* :meth:`synthesize` 内部通过 ODE 求解生成 mel，再经 HiFi-GAN 解码
* :meth:`synthesize_stream` 使用 Chunk-aware ODE 求解（非逐 token 流式）

Examples
--------
>>> backend = CosyVoiceBackend(model_path="/data/cosyvoice")
>>> backend.load(device="cuda", dtype="float16")
>>> audio = backend.synthesize("你好，世界", speaker=None, language="zh")

Notes
-----
CosyVoice 模型遵循 **Apache-2.0** 许可。输出采样率为 24000Hz
（与 HuggingFace 仓库 ``FunAudioLLM/CosyVoice2-0.5B`` 的
``cosyvoice2.yaml`` 一致）。

## 类属性

| 名称 | 值 |
|---|---|
| `name` | `'cosyvoice'` |

## 方法

### `__init__(self, model_path: 'str', llm_model: 'str' = 'Qwen/Qwen2.5-1.5B-Instruct', speech_tokenizer_model: 'str | None' = None, speaker_encoder_model: 'str' = 'campp', hifi_gan_path: 'str | None' = None, num_ode_steps: 'int' = 10, ode_solver: 'str' = 'euler', language: 'str' = 'zh', streaming_enabled: 'bool' = True, chunk_size_frames: 'int' = 150, chunk_overlap_frames: 'int' = 16, scheduler: 'Any' = None, repo_id: 'str | None' = None) -> 'None'`

初始化 CosyVoice 后端。

Parameters
----------
model_path : str
    CosyVoice 模型根路径。本地不存在时，将通过
    :class:`HFModelManager` 从 ``repo_id`` / 默认仓库下载。
llm_model : str
    文本理解 LLM 模型名称或路径（HF 布局未找到 ``CosyVoice-BlankEN``
    时的回退）。
speech_tokenizer_model : str | None
    语音 Tokenizer 权重路径；``None`` 时按 HF 布局查找
    ``speech_tokenizer_v2.onnx``，找不到再回退到旧目录
    ``model_path/speech_tokenizer``。
speaker_encoder_model : str
    说话人编码器类型，默认 ``"campp"``。
hifi_gan_path : str | None
    HiFi-GAN 权重路径；``None`` 时按 HF 布局查找 ``hift.pt``，
    找不到再回退到旧目录 ``model_path/hifi_gan``。
num_ode_steps : int
    ODE 求解步数，默认 ``10``。
ode_solver : str
    ODE 求解器，``"euler"`` / ``"midpoint"`` / ``"rk4"``。
language : str
    默认语言代码。
streaming_enabled : bool
    是否启用流式合成。
chunk_size_frames : int
    流式 chunk 大小（帧数），默认 ``150``。
chunk_overlap_frames : int
    chunk 重叠帧数，默认 ``16``。
scheduler : Any
    显存调度器实例。
repo_id : str | None
    HuggingFace 仓库 ID（如 ``"FunAudioLLM/CosyVoice2-0.5B"``）。
    ``None`` 时使用后端默认仓库 ``cosyvoice``。

### `benchmark_ode_steps(self, text: 'str', steps_list: 'list[int] | None' = None) -> 'dict[int, dict[str, float]]'`

测试不同 ODE 步数的质量和速度。

Parameters
----------
text : str
    测试文本。
steps_list : list[int] | None
    要测试的步数列表，默认 ``[5, 10, 20, 50]``。

Returns
-------
dict[int, dict[str, float]]
    ``{步数: {"time": 耗时秒, "mel_std": mel标准差}}``。

### `clone_voice(self, audio: 'AudioData | str', text: 'str', language: 'str' = 'zh', **kwargs: 'Any') -> 'AudioData'`

语音克隆的便捷方法。

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

Returns
-------
AudioData
    克隆语音结果。

### `clone_voice_stream(self, audio: 'AudioData | str', text: 'str', language: 'str' = 'zh', **kwargs: 'Any') -> 'Iterator[AudioData]'`

流式语音克隆。

### `describe(self) -> 'TTSBackendSpec'`

返回本后端的规格描述。

### `extract_speaker(self, audio: 'AudioData | str') -> 'dict[str, Any]'`

从音频中提取说话人特征。

Parameters
----------
audio : AudioData | str
    参考音频。

Returns
-------
dict[str, Any]
    包含 ``ref_speech_tokens`` 和 ``speaker_embedding`` 的字典。

### `list_speakers(self) -> 'list[str]'`

返回预计算的说话人列表。

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

### `save_speaker(self, name: 'str', audio: 'AudioData | str') -> 'None'`

提取并保存说话人特征到本地。

### `set_ode_params(self, num_steps: 'int', solver: 'str' = 'euler') -> 'None'`

运行时修改 ODE 参数。

用于在质量和速度之间做权衡调整。

Parameters
----------
num_steps : int
    ODE 求解步数（5=最快, 10=推荐, 20=高质量）。
solver : str
    ODE 求解器：``"euler"`` / ``"midpoint"`` / ``"rk4"``。

### `synthesize(self, text: 'str', speaker: 'str | None' = None, language: 'str' = 'zh', speed: 'float' = 1.0, **kwargs: 'Any') -> 'AudioData'`

阻塞式合成完整语音。

完整流程：

1. 文本校验。
2. CosyVoiceTokenizer.preprocess(text)
3. token_ids = CosyVoiceTokenizer.tokenize(text, language)
4. text_feats = LLM(token_ids) → text_hidden_states
5. 参考音频处理（如果有 speaker）
6. mel = FlowMatchingModel.generate(text_feats, speaker_info, ...)
7. waveform = HiFiGanVocoder.decode(mel)
8. 如果 speed != 1.0，做时间拉伸
9. 返回 AudioData(waveform, sample_rate=24000)

Parameters
----------
text : str
    待合成文本。
speaker : str | None
    说话人名称或参考音频路径。
language : str
    语言代码。
speed : float
    语速倍率。
**kwargs : Any
    额外参数（``num_ode_steps`` / ``ode_solver`` 等）。

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

分块流式合成。

使用 Chunk-aware ODE 求解策略：
1. 将目标 mel 分为多个 chunk（每个 ~150 帧 ≈ 1.74 秒）
2. 每个 chunk 独立做 ODE 求解
3. 每个 chunk 完成后经 HiFi-GAN 解码为波形
4. 通过 StreamAdapter 缓冲输出

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
