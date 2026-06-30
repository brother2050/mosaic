# `ChatTTSBackend`

**模块**：`mosaic.nodes.audio.tts_backends.implementations.chattts_backend`
**继承**：`TTSBackend`

## 描述

ChatTTS TTS 后端。

将 ChatTTS 的文本前端（:class:`ChatTokenizer`）、声学模型
（:class:`LlamaARModel`）、复合声码器（DVAE + Vocos）与流式适配器
（:class:`StreamAdapter`）组装为统一的 :class:`TTSBackend`，支持中英文
阻塞合成与流式合成，并通过随机种子生成说话人嵌入。

生命周期
--------
1. 构造后端实例（``is_loaded=False``）。
2. 调用 :meth:`load` 加载四层管线（``is_loaded=True``）。
3. 调用 :meth:`synthesize` / :meth:`synthesize_stream` 合成语音。
4. 调用 :meth:`unload` 释放资源。

Examples
--------
>>> backend = ChatTTSBackend(model_path="/data/chattts")
>>> backend.load(device="cuda", dtype="float16")
>>> audio = backend.synthesize("你好，世界", speaker=None, language="zh")

Notes
-----
ChatTTS 模型遵循 **CC BY-NC 4.0** 许可，仅供非商业用途。

## 类属性

| 名称 | 值 |
|---|---|
| `name` | `'chattts'` |

## 方法

### `__init__(self, model_path: 'str', vocos_path: 'str | None' = None, num_vq: 'int' = 4, language: 'str' = 'zh', use_flash_attention: 'bool' = True, streaming_enabled: 'bool' = True, stream_batch: 'int' = 24, scheduler: 'Any' = None, repo_id: 'str | None' = None) -> 'None'`

初始化 ChatTTS 后端。

Parameters
----------
model_path : str
    ChatTTS 模型目录路径。若本地目录不存在或为空，将通过
    :class:`HFModelManager` 从 HuggingFace 仓库下载（默认
    ``2Noise/ChatTTS``）。HF 仓库布局为 ``config/`` (YAML 配置)
    + ``asset/`` (权重与 tokenizer)。
vocos_path : str | None, default None
    Vocos 权重路径；``None`` 时自动从 ``asset/Vocos.safetensors``
    查找。提供自定义路径时优先使用。
num_vq : int, default 4
    VQ 码本组数。若 ``config/gpt.yaml`` 存在，以配置值为准。
language : str, default "zh"
    默认语言代码。
use_flash_attention : bool, default True
    声学模型是否使用 Flash Attention 加速。
streaming_enabled : bool, default True
    是否启用流式合成（构建 Layer 4 流式适配器）。
stream_batch : int, default 24
    流式生成每次 yield 的 token 数。
scheduler : Any
    显存调度器实例，``None`` 使用全局单例。透传给
    :meth:`TTSBackend.__init__`。
repo_id : str | None, default None
    HuggingFace 仓库 ID（如 ``"2Noise/ChatTTS"``）。``None`` 时
    使用 :attr:`HFModelManager.DEFAULT_REPOS` 中 ``"chattts"`` 对应
    的默认仓库。

### `describe(self) -> 'TTSBackendSpec'`

返回本后端的规格描述。

### `list_speakers(self) -> 'list[str]'`

返回内置的说话人列表。

ChatTTS 通过随机种子（Seed）生成说话人嵌入，本身不内置固定说话人。
此处返回常用的种子标识列表，用户可结合 :meth:`set_seed` 与
:meth:`sample_random_speaker` 生成对应的说话人嵌入字符串，再作为
``speaker`` 参数传入 :meth:`synthesize`。

Returns
-------
list[str]
    常用种子标识列表，如 ``["seed_2", "seed_222", ...]``。

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

### `sample_random_speaker(self) -> 'str'`

随机采样一个说话人嵌入并编码为字符串。

采用高斯分布采样：``spk = randn * std + mean``，其中 ``mean`` / ``std``
从模型路径下的 ``asset/spk_stat.pt`` 加载（若存在），否则使用默认值。
采样后经 ``float16`` 量化、LZMA2 压缩、Base16384 编码为字符串，与
:meth:`ChatTokenizer.encode_speaker` 的解码流程对称。

Returns
-------
str
    Base16384 + LZMA2 编码的说话人嵌入字符串，可直接作为 ``speaker``
    参数传入 :meth:`synthesize`。

Raises
------
ImportError
    ``torch`` 未安装。

### `set_seed(self, seed: 'int') -> 'None'`

设置随机种子（影响说话人采样与声学模型生成）。

Parameters
----------
seed : int
    随机种子。

### `synthesize(self, text: 'str', speaker: 'str | None' = None, language: 'str' = 'zh', speed: 'float' = 1.0, **kwargs: 'Any') -> 'AudioData'`

阻塞式合成完整语音。

完整流程：

1. 检查 ``is_loaded``。
2. 文本校验。
3. 提取韵律提示 ``prosody_prompt``（从 ``kwargs`` 中移除，避免透传到
   声学模型）。
4. 文本清洗（``ChatTokenizer.preprocess``）。
5. 韵律标记插入（``ChatTokenizer.insert_prosody_tokens``）。
6. 分词（``ChatTokenizer.tokenize``）→ token_ids。
7. 说话人嵌入解码（``ChatTokenizer.encode_speaker``）。
8. 合并推理参数（``spec.default_params`` 与 ``kwargs``）。
9. 声学模型生成（``LlamaARModel.generate``）→ audio_codes。
10. 复合声码器解码（``_CompositeVocoder.decode``）→ waveform。
11. 构造 :class:`AudioData` 返回。

.. note::
   步骤 4-6 由 ``ChatTokenizer.tokenize`` 内部统一完成（清洗在前、
   韵律在后），此处透传 ``prosody_prompt`` 与 ``speaker_id`` 由其
   统一处理，以保证韵律特殊标记不会被重复预处理破坏。

Parameters
----------
text : str
    待合成文本。
speaker : str | None
    说话人标识（Base16384 编码的说话人嵌入字符串）；``None`` 使用
    空说话人。
language : str
    语言代码，默认 ``"zh"``。
speed : float
    语速倍率（记录于元数据；ChatTTS 通过韵律标记控制语速）。
**kwargs : Any
    额外参数，包括 ``prosody_prompt``（韵律提示）以及透传给声学模型
    的 ``temperature`` / ``top_p`` / ``top_k`` / ``max_new_tokens``
    等（覆盖 ``spec.default_params``）。

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

1. 前处理同 :meth:`synthesize`（步骤 1-8）。
2. 创建 :class:`StreamSession` 流式会话。
3. 对声学模型的流式输出逐块迭代：

   a. ``LlamaARModel.generate_stream`` → audio_codes_chunk
   b. 复合声码器 ``decode_chunk`` → waveform_chunk
   c. ``StreamSession.push(waveform_chunk)``
   d. ``StreamSession.pop()`` → yield :class:`AudioData`

4. 冲刷缓冲区中剩余数据。

若后端不支持流式（``streaming_enabled=False``），回退为
:meth:`synthesize` 一次性返回完整结果。

Parameters
----------
text : str
    待合成文本。
speaker : str | None
    说话人标识。
language : str
    语言代码，默认 ``"zh"``。
speed : float
    语速倍率。
chunk_size : int
    每个音频块的目标采样数，默认 ``4096``。
**kwargs : Any
    额外参数，同 :meth:`synthesize`。

Yields
------
AudioData
    逐块音频数据，``metadata`` 中 ``streaming=True``。

### `unload(self) -> 'None'`

释放模型权重与四层管线资源。
