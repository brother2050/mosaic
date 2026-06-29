# Mosaic 插件开发指南

> 编写自定义节点、注册插件、发布到 PyPI 的完整流程。

## 目录

- [快速创建插件](#快速创建插件)
- [插件目录结构](#插件目录结构)
- [实现自定义 Node](#实现自定义-node)
- [注册插件的三种机制](#注册插件的三种机制)
- [测试插件](#测试插件)
- [发布插件到 PyPI](#发布插件到-pypi)
- [完整示例：情感分析节点](#完整示例情感分析节点)
- [完整示例：自定义 TTS 后端](#完整示例自定义-tts-后端)

---

## 快速创建插件

使用 CLI 模板快速生成节点骨架：

```bash
mosaic create-node --domain text --name sentiment-analyzer
```

输出：

```
节点模板已生成: ['/path/to/my_nodes/sentiment_analyzer.py', '/path/to/my_nodes/__init__.py', '/path/to/my_nodes/test_sentiment_analyzer.py', '/path/to/my_nodes/README.md']
```

生成的文件位于 `--output` 指定的目录（默认 `./my_nodes/`），包含节点代码、包初始化文件、测试骨架和说明文档。

模板文件内容：

```python
# mosaic/nodes/text/sentiment_analyzer.py
"""情感分析节点。"""
from __future__ import annotations

from typing import Any

from mosaic.core.node import Node, NodeSpec
from mosaic.core.registry import registry
from mosaic.core.types import MosaicData


@registry.register
class SentimentAnalyzer(Node):
    """情感分析节点。
    
    一句话描述：分析文本的情感倾向。
    """
    name: str = "sentiment-analyzer"
    domain: str = "text"
    description: str = "Analyze text sentiment (positive/negative/neutral)."
    version: str = "0.1.0"
    input_types: list[str] = ["text"]
    output_types: list[str] = ["text", "mosaic"]
    
    def __init__(self, model: str = "default", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model = model
    
    def run(self, input_data: MosaicData) -> MosaicData:
        text = input_data.get("text", "")
        # TODO: 实现情感分析逻辑
        return input_data.set("sentiment", "neutral")
    
    def describe(self) -> NodeSpec:
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=self.input_types,
            output_types=self.output_types,
        )
```

---

## 插件目录结构

Mosaic 插件的推荐目录结构：

```
my-mosaic-plugin/
├── pyproject.toml
├── README.md
├── LICENSE
├── my_plugin/
│   ├── __init__.py
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── text/
│   │   │   ├── __init__.py
│   │   │   └── sentiment_analyzer.py
│   │   └── audio/
│   │       └── custom_tts.py
│   └── backends/
│       └── custom_tts_backend.py
└── tests/
    ├── __init__.py
    ├── test_sentiment_analyzer.py
    └── test_custom_tts.py
```

---

## 实现自定义 Node

### 基础节点

最简节点只需要实现 `run()`：

```python
from mosaic.core.node import Node
from mosaic.core.registry import registry
from mosaic.core.types import MosaicData


@registry.register
class UppercaseNode(Node):
    name: str = "uppercase"
    domain: str = "text"
    description: str = "Convert text to uppercase."
    
    def run(self, input_data: MosaicData) -> MosaicData:
        text = input_data.get("text", "")
        return input_data.set("text", text.upper())
```

### 加载模型的节点

涉及 ML 模型的节点需要实现 `load()` / `unload()`：

```python
import os
from mosaic.core.node import Node
from mosaic.core.registry import registry
from mosaic.core.types import MosaicData


@registry.register
class SentimentAnalyzer(Node):
    name: str = "sentiment-analyzer"
    domain: str = "text"
    description: str = "Sentiment analysis with HF transformers."
    input_types: list[str] = ["text"]
    output_types: list[str] = ["text", "mosaic"]
    
    def __init__(
        self,
        model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model_name = model
        self._device = device
        self._pipeline = None  # 延迟加载
    
    def load(self) -> None:
        """加载模型（由 Scheduler 调用）"""
        from transformers import pipeline
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=self._model_name,
            device=self._device,
        )
        self._loaded = True
    
    def unload(self) -> None:
        """释放资源"""
        self._pipeline = None
        self._loaded = False
    
    def run(self, input_data: MosaicData) -> MosaicData:
        # 触发按需加载
        from mosaic.core.scheduler import get_scheduler
        get_scheduler().ensure_loaded(self)
        
        text = input_data.get("text", "")
        if not text:
            raise ValueError("SentimentAnalyzer requires 'text' field")
        
        result = self._pipeline(text)[0]
        return input_data.set("sentiment", result["label"].lower()) \
                         .set("score", result["score"])
    
    def describe(self) -> NodeSpec:
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            model_info={"name": self._model_name, "vram_gb": 1.0},
        )
```

### 继承域基类

如需复用域的共性逻辑（如视频域的帧处理），继承域基类：

```python
from mosaic.nodes.video._base import BaseVideoNode
from mosaic.core.registry import registry


@registry.register
class MyVideoNode(BaseVideoNode):
    name: str = "my-video"
    domain: str = "video"
    
    def _load_model(self):
        # 复用 BaseVideoNode 的设备/精度解析
        from diffusers import MyPipeline
        self._pipeline = MyPipeline.from_pretrained(
            self._model_name,
            torch_dtype=self._resolve_dtype(),
        ).to(self._resolve_device())
    
    def run(self, input_data):
        # 复用 BaseVideoNode 的调度/事件机制
        self._scheduler.ensure_loaded(self)
        self._emit_start()
        # ... 推理逻辑 ...
        self._emit_complete()
        return input_data
```

---

## 注册插件的三种机制

### 机制 1：entry_points（PyPI 插件）

在第三方包的 `pyproject.toml` 中声明：

```toml
[project]
name = "mosaic-plugin-sentiment"
version = "0.1.0"
dependencies = ["mosaic>=0.1.0", "transformers>=4.40"]

[project.entry-points."mosaic.nodes"]
sentiment-analyzer = "mosaic_plugin_sentiment.nodes:SentimentAnalyzer"
```

安装时自动注册：

```bash
pip install mosaic-plugin-sentiment
# SentimentAnalyzer 已自动出现在 registry 中
```

`mosaic` 启动时会扫描所有 `mosaic.nodes` entry points。

### 机制 2：装饰器（应用代码内）

最简单，import 时自动注册：

```python
from mosaic.core.registry import registry

@registry.register
class MyNode(Node): ...
```

适合：

- 应用内节点
- 内部模块
- 单仓库项目

### 机制 3：目录扫描（运行时）

运行时从任意目录加载：

```python
from mosaic.core.plugin import PluginManager

pm = PluginManager()
pm.discover_directory("/path/to/custom_nodes")

# 列出已加载
print(pm.list_nodes())
```

适合：

- 不重启应用加载新节点
- 用户自定义扩展目录
- 第三方自定义节点（不发布到 PyPI）

### 同时使用多种

```python
# 主项目内置节点用装饰器
@registry.register
class BuiltinNode(Node): ...

# 第三方插件用 entry_points
# （pip install 时自动加载）

# 用户扩展用目录扫描
pm.discover_directory("./my_custom_nodes")
```

---

## 测试插件

Mosaic 的测试约定：

```
tests/
├── phase1/  # 核心框架
├── phase2/  # 节点基础
├── phase3/  # 域节点
├── phase4/  # 视频/图像
├── phase5/  # 音频/TTS
├── phase6/  # 高级特性
├── phase7/  # 集成
└── tts/     # TTS 后端
```

新建节点测试时，先看一下对应阶段的 conftest.py：

```python
# tests/phase3/test_sentiment_analyzer.py
"""Phase 3 SentimentAnalyzer 节点测试。"""

import sys
sys.path.insert(0, "/workspace/mosaic")

import pytest
from mosaic.core.types import MosaicData
from mosaic.nodes.text.sentiment_analyzer import SentimentAnalyzer


class TestSentimentAnalyzer:
    """情感分析节点测试。"""
    
    @pytest.fixture(autouse=True)
    def _setup(self, cpu_scheduler):
        self.node = SentimentAnalyzer(
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            scheduler=cpu_scheduler,
        )
        # 不实际 load()，用 mock
    
    def test_returns_sentiment_label(self):
        """正面文本应被识别为 positive。"""
        # 模拟 pipeline
        with patch.object(self.node, "_pipeline") as mock_pipe:
            mock_pipe.return_value = [{"label": "positive", "score": 0.95}]
            
            result = self.node.run(MosaicData(text="I love this product!"))
            assert result.get("sentiment") == "positive"
    
    def test_requires_text(self):
        """缺少 text 应抛 ValueError。"""
        with pytest.raises(ValueError, match="text"):
            self.node.run(MosaicData())
```

### 测试辅助 fixtures

`tests/phase*/conftest.py` 提供了常用 fixture：

- `cpu_scheduler` — 不依赖 GPU 的调度器
- `event_bus` — 事件总线
- `mock_diffusers` — 自动注入 mock diffusers

---

## 发布插件到 PyPI

### 1. 准备 `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mosaic-plugin-sentiment"
version = "0.1.0"
description = "Sentiment analysis plugin for Mosaic"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "Apache-2.0"}
dependencies = [
    "mosaic>=0.1.0",
    "transformers>=4.40",
    "torch>=2.0",
]

[project.entry-points."mosaic.nodes"]
sentiment-analyzer = "mosaic_plugin_sentiment.nodes:SentimentAnalyzer"
```

### 2. 构建

```bash
pip install build twine
python -m build
```

生成 `dist/` 下的 `*.whl` 和 `*.tar.gz`。

### 3. 上传

```bash
# TestPyPI（先在这里测试）
twine upload --repository testpypi dist/*

# 正式 PyPI
twine upload dist/*
```

### 4. 验证

```bash
pip install mosaic-plugin-sentiment
python -c "from mosaic.core.registry import registry; print('sentiment-analyzer' in registry)"
# True
```

### 5. 命名约定

| 类型 | 命名 |
|---|---|
| PyPI 包名 | `mosaic-plugin-<name>` 或 `mosaic-<name>` |
| Python 模块名 | `mosaic_plugin_<name>` |
| 节点名 | `<kebab-case>` |

---

## 完整示例：情感分析节点

从零创建一个完整的情感分析插件。

### 目录结构

```
mosaic-plugin-sentiment/
├── pyproject.toml
├── README.md
├── LICENSE
├── mosaic_plugin_sentiment/
│   ├── __init__.py
│   └── nodes.py
└── tests/
    └── test_sentiment.py
```

### 1. `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "mosaic-plugin-sentiment"
version = "0.1.0"
description = "Sentiment analysis node for Mosaic"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "Apache-2.0"}
authors = [{name = "Your Name", email = "you@example.com"}]
dependencies = [
    "mosaic>=0.1.0",
    "transformers>=4.40",
]

[project.entry-points."mosaic.nodes"]
sentiment-analyzer = "mosaic_plugin_sentiment.nodes:SentimentAnalyzer"
```

### 2. `mosaic_plugin_sentiment/__init__.py`

```python
"""Sentiment analysis plugin for Mosaic."""
__version__ = "0.1.0"
```

### 3. `mosaic_plugin_sentiment/nodes.py`

```python
"""情感分析节点实现。"""
from __future__ import annotations

from typing import Any

from mosaic.core.node import Node, NodeSpec
from mosaic.core.registry import registry
from mosaic.core.scheduler import get_scheduler
from mosaic.core.types import MosaicData


@registry.register
class SentimentAnalyzer(Node):
    """情感分析节点。
    
    使用 HuggingFace transformers 的预训练模型进行情感分类。
    支持多语言（取决于模型）。
    
    Parameters
    ----------
    model : str
        HF 模型 ID，默认 ``"cardiffnlp/twitter-roberta-base-sentiment-latest"``。
    device : str
        推理设备，默认 ``"cuda"``。
    
    Examples
    --------
    >>> sa = SentimentAnalyzer()
    >>> result = sa(MosaicData(text="I love this!"))
    >>> result.get("sentiment")
    'positive'
    """
    name: str = "sentiment-analyzer"
    domain: str = "text"
    description: str = "Analyze text sentiment using HF transformers."
    version: str = "0.1.0"
    input_types: list[str] = ["text", "mosaic"]
    output_types: list[str] = ["text", "mosaic"]
    
    def __init__(
        self,
        model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: str = "cuda",
        scheduler=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name = model
        self._device = device
        self._scheduler = scheduler or get_scheduler()
        self._pipeline = None
    
    def load(self) -> None:
        """加载 transformers pipeline。"""
        from transformers import pipeline  # type: ignore
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=self._model_name,
            device=self._device if self._device == "cpu" else int(self._device.split(":")[-1]) if self._device.startswith("cuda") else self._device,
        )
        self._loaded = True
    
    def unload(self) -> None:
        """释放 pipeline。"""
        self._pipeline = None
        self._loaded = False
    
    def run(self, input_data: MosaicData) -> MosaicData:
        """执行情感分析。
        
        Parameters
        ----------
        input_data : MosaicData
            必须包含 ``text`` (str)。
        
        Returns
        -------
        MosaicData
            包含 ``sentiment`` (str) 和 ``score`` (float)。
        """
        self._scheduler.ensure_loaded(self)
        
        text = input_data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("SentimentAnalyzer requires non-empty 'text'.")
        
        result = self._pipeline(text)[0]
        return input_data.set("sentiment", result["label"].lower()) \
                         .set("score", float(result["score"]))
    
    def describe(self) -> NodeSpec:
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=self.input_types,
            output_types=self.output_types,
            model_info={
                "name": self._model_name,
                "vram_gb": 1.0,
                "license": "Various (model-dependent)",
            },
        )
```

### 4. `tests/test_sentiment.py`

```python
import sys
sys.path.insert(0, "/workspace/mosaic")

from unittest.mock import patch, MagicMock
import pytest

from mosaic.core.types import MosaicData
from mosaic_plugin_sentiment.nodes import SentimentAnalyzer


class TestSentimentAnalyzer:
    @pytest.fixture(autouse=True)
    def _setup(self, cpu_scheduler):
        self.node = SentimentAnalyzer(scheduler=cpu_scheduler)
        self.node._pipeline = MagicMock()
    
    def test_positive_text(self):
        self.node._pipeline.return_value = [{"label": "POSITIVE", "score": 0.95}]
        result = self.node.run(MosaicData(text="Great!"))
        assert result.get("sentiment") == "positive"
        assert result.get("score") == 0.95
    
    def test_negative_text(self):
        self.node._pipeline.return_value = [{"label": "NEGATIVE", "score": 0.88}]
        result = self.node.run(MosaicData(text="Terrible!"))
        assert result.get("sentiment") == "negative"
    
    def test_missing_text_raises(self):
        with pytest.raises(ValueError, match="text"):
            self.node.run(MosaicData())
```

### 5. 集成到 Pipeline

```python
# 安装：pip install mosaic-plugin-sentiment
from mosaic import Pipeline
from mosaic_plugin_sentiment.nodes import SentimentAnalyzer
from mosaic.nodes.text import TextGenerator

pipe = TextGenerator() | SentimentAnalyzer()
result = pipe.run(prompt="Write a product review, then analyze it")
print(result.get("sentiment"))
```

---

## 完整示例：自定义 TTS 后端

实现一个简单的 TTS 后端（如 eSpeak-NG 或自训练模型）：

### 1. 实现 Backend 基类

```python
# my_tts_backend.py
"""自定义 TTS 后端：基于 eSpeak-NG 的轻量 TTS。"""
from __future__ import annotations

import subprocess
import tempfile
import wave
import numpy as np
from pathlib import Path
from typing import Iterator

from mosaic.core.types import AudioData
from mosaic.nodes.audio.tts_backends.base import TTSBackend, TTSBackendMeta


class EspeakBackend(TTSBackend):
    """基于 eSpeak-NG 的轻量 TTS 后端。
    
    特点：
    - CPU 运行，无需 GPU
    - 极小模型 (~10MB)
    - 支持 100+ 语言
    - 适合快速原型和低资源场景
    - 不支持流式
    """
    
    META = TTSBackendMeta(
        name="espeak",
        sample_rate=22050,
        supports_streaming=False,
        supports_cloning=False,
        license="GPL-3.0",
    )
    
    def __init__(self, voice: str = "en", speed: int = 175):
        self._voice = voice
        self._speed = speed
    
    def load(self) -> None:
        # 检查 eSpeak 是否安装
        try:
            subprocess.run(["espeak-ng", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "espeak-ng not found. Install with:\n"
                "  apt install espeak-ng (Linux)\n"
                "  brew install espeak (macOS)"
            ) from exc
        self._loaded = True
    
    def unload(self) -> None:
        self._loaded = False
    
    def synthesize(self, text: str, language: str = "en", **kwargs) -> AudioData:
        self._ensure_loaded()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            subprocess.run([
                "espeak-ng",
                "-v", language,
                "-s", str(self._speed),
                "-w", tmp_path,
                text,
            ], check=True, capture_output=True)
            
            # 读取 wav
            with wave.open(tmp_path, "rb") as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            
            return AudioData(
                waveform=audio,
                sample_rate=sr,
                metadata={"text": text, "language": language, "backend": "espeak"},
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def synthesize_stream(self, text: str, **kwargs) -> Iterator[AudioData]:
        # 不支持流式，一次性返回
        yield self.synthesize(text, **kwargs)
```

### 2. 注册到 TTS 节点

```python
# 在 mosaic_tts_registry.py 中注册
from mosaic.nodes.audio import TTS
from my_tts_backend import EspeakBackend

TTS.register_backend("espeak", EspeakBackend)
```

### 3. 使用

```python
from mosaic.nodes.audio import TTS

tts = TTS(backend="espeak", voice="zh")
audio = tts.run({"text": "你好世界", "language": "zh"}).get("audio")
audio.save("hello.wav")
```

---

## 下一步

- [节点参考手册](nodes-reference.md) — 内置节点 API
- [架构设计](architecture.md) — 核心模块
- [示例代码](../examples/) — 实际应用
