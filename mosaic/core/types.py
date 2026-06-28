# mosaic/core/types.py
"""Mosaic 统一数据类型定义。

本模块定义了节点之间传递的所有数据类型。所有数据类型均继承自
``MosaicData``，支持字典式访问、序列化/反序列化与类型校验。

设计要点
--------
* ``MosaicData`` 是顶层容器，行为类似 ``dict``，支持任意 key-value。
* 各具体数据类型（``TextData``/``ImageData`` 等）在其基础上增加强类型字段。
* ``to_dict`` / ``from_dict`` 负责序列化。为保持可移植性，``PIL.Image``
  被编码为 base64 PNG 字符串，``numpy.ndarray`` 被转换为带形状/dtype
  信息的嵌套列表。``PIL`` 与 ``numpy`` 采用惰性导入，缺失时模块仍可加载。
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type

__all__ = [
    "MosaicData",
    "TextData",
    "ImageData",
    "AudioData",
    "VideoData",
    "SubtitleData",
    "DocumentData",
    "RagQueryResult",
    "DATA_TYPE_REGISTRY",
    "data_from_dict",
]


# ---------------------------------------------------------------------------
# 内部辅助：惰性导入 PIL / numpy，避免硬依赖
# ---------------------------------------------------------------------------
def _import_pil() -> Any:
    """惰性导入 PIL.Image，缺失时抛出带提示的 ImportError。"""
    try:
        from PIL import Image  # type: ignore
        return Image
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise ImportError(
            "Pillow is required for image serialization. "
            "Install it via `pip install Pillow`."
        ) from exc


def _import_numpy() -> Any:
    """惰性导入 numpy，缺失时抛出带提示的 ImportError。"""
    try:
        import numpy  # type: ignore
        return numpy
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise ImportError(
            "numpy is required for array serialization. "
            "Install it via `pip install numpy`."
        ) from exc


def _image_to_b64(image: Any) -> str:
    """将 PIL.Image 编码为 ``b64:<png-data>`` 字符串。"""
    Image = _import_pil()
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image)!r}.")
    buf = io.BytesIO()
    fmt = image.format if image.format else "PNG"
    image.save(buf, format=fmt)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"b64:{fmt}:{encoded}"


def _b64_to_image(token: str) -> Any:
    """将 ``b64:<fmt>:<data>`` 字符串还原为 PIL.Image。"""
    Image = _import_pil()
    if not token.startswith("b64:"):
        raise ValueError("Invalid image token, expected 'b64:<fmt>:<data>'.")
    _, fmt, encoded = token.split(":", 2)
    raw = base64.b64decode(encoded.encode("ascii"))
    return Image.open(io.BytesIO(raw))


def _array_to_dict(array: Any) -> Dict[str, Any]:
    """将 numpy.ndarray 序列化为带元数据的字典。"""
    np = _import_numpy()
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(array)!r}.")
    return {
        "__ndarray__": True,
        "data": array.tolist(),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }


def _dict_to_array(payload: Dict[str, Any]) -> Any:
    """将带元数据的字典还原为 numpy.ndarray。"""
    np = _import_numpy()
    data = payload["data"]
    dtype = payload.get("dtype", "float32")
    return np.array(data, dtype=dtype).reshape(payload.get("shape", -1))


def _serialize_value(value: Any) -> Any:
    """递归序列化单个值，处理 PIL/numpy 等特殊类型。"""
    # PIL.Image
    try:
        Image = _import_pil()
        if isinstance(value, Image.Image):
            return {"__pil_image__": True, "encoded": _image_to_b64(value)}
    except ImportError:
        pass

    # numpy.ndarray
    try:
        np = _import_numpy()
        if isinstance(value, np.ndarray):
            return _array_to_dict(value)
    except ImportError:
        pass

    # 元组：用标记保留元组语义（JSON 不区分 list/tuple）
    if isinstance(value, tuple):
        return {"__tuple__": True, "items": [_serialize_value(v) for v in value]}

    # 列表：递归处理元素
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]

    # 字典：递归处理
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}

    return value


def _deserialize_value(value: Any) -> Any:
    """递归反序列化单个值，还原 PIL/numpy/tuple 等特殊类型。"""
    if isinstance(value, dict):
        if value.get("__tuple__"):
            return tuple(_deserialize_value(v) for v in value["items"])
        if value.get("__pil_image__"):
            return _b64_to_image(value["encoded"])
        if value.get("__ndarray__"):
            return _dict_to_array(value)
        return {k: _deserialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deserialize_value(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# MosaicData — 顶层字典式容器
# ---------------------------------------------------------------------------
class MosaicData:
    """所有节点数据的顶层容器。

    行为类似 ``dict``，支持任意 key-value，同时提供序列化与类型校验能力。
    所有具体数据类型（``TextData`` 等）均继承本类。

    Parameters
    ----------
    **kwargs:
        任意键值对，将存入内部数据字典。
    """

    #: 数据类型标识，用于反序列化时分发到正确的子类。
    data_type: str = "mosaic"

    def __init__(self, **kwargs: Any) -> None:
        self._data: Dict[str, Any] = dict(kwargs)

    # -- 字典式协议 --------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MosaicData):
            return self._data == other._data
        if isinstance(other, dict):
            return self._data == other
        return NotImplemented

    def keys(self) -> Any:
        """返回所有键的视图。"""
        return self._data.keys()

    def values(self) -> Any:
        """返回所有值的视图。"""
        return self._data.values()

    def items(self) -> Any:
        """返回所有 (键, 值) 对的视图。"""
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        """安全取值，键不存在时返回默认值。"""
        return self._data.get(key, default)

    def update(self, other: Dict[str, Any]) -> None:
        """用 ``other`` 中的键值对更新当前数据。"""
        self._data.update(other)

    # -- 序列化 ------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """将数据序列化为纯字典（可 JSON 化）。

        ``PIL.Image`` 编码为 base64 PNG，``numpy.ndarray`` 转为带元数据的
        嵌套列表，确保返回值可被 ``json`` 等标准序列化器处理。
        """
        payload: Dict[str, Any] = {
            "__data_type__": self.data_type,
        }
        for key, value in self._data.items():
            payload[key] = _serialize_value(value)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MosaicData":
        """从字典反序列化为数据实例。

        若字典中包含 ``__data_type__`` 字段，将分发到对应子类。
        """
        dtype = data.get("__data_type__", cls.data_type)
        target_cls: Type[MosaicData] = DATA_TYPE_REGISTRY.get(dtype, cls)
        # 过滤掉元信息键
        clean = {
            k: _deserialize_value(v)
            for k, v in data.items()
            if k != "__data_type__"
        }
        return target_cls(**clean)

    # -- 类型校验 ----------------------------------------------------------
    @classmethod
    def validate(cls, data: "MosaicData") -> bool:
        """校验给定数据是否符合本类型约束。

        基类实现仅检查类型归属；子类可覆写以增加字段级校验。
        """
        return isinstance(data, cls)

    # -- 表示 --------------------------------------------------------------
    def __repr__(self) -> str:
        items_str = ", ".join(f"{k}={v!r}" for k, v in self._data.items())
        return f"{self.__class__.__name__}({items_str})"


# ---------------------------------------------------------------------------
# TextData — 文本数据
# ---------------------------------------------------------------------------
class TextData(MosaicData):
    """文本数据。

    Parameters
    ----------
    content:
        文本内容。
    language:
        语言代码（如 ``"zh"``、``"en"``），默认 ``"auto"``。
    metadata:
        附加元数据。
    """

    data_type = "text"

    def __init__(
        self,
        content: str = "",
        language: str = "auto",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            content=content,
            language=language,
            metadata=metadata or {},
            **kwargs,
        )

    @property
    def content(self) -> str:
        """文本内容。"""
        return self._data["content"]

    @property
    def language(self) -> str:
        """语言代码。"""
        return self._data["language"]

    @property
    def metadata(self) -> Dict[str, Any]:
        """附加元数据。"""
        return self._data["metadata"]

    @classmethod
    def validate(cls, data: "MosaicData") -> bool:
        """校验文本数据：必须含字符串 ``content``。"""
        if not isinstance(data, TextData):
            return False
        return isinstance(data.get("content"), str)


# ---------------------------------------------------------------------------
# ImageData — 图像数据
# ---------------------------------------------------------------------------
class ImageData(MosaicData):
    """图像数据。

    Parameters
    ----------
    image:
        ``PIL.Image.Image`` 实例。
    size:
        ``(width, height)`` 元组。
    metadata:
        附加元数据。
    """

    data_type = "image"

    def __init__(
        self,
        image: Any = None,
        size: Optional[Tuple[int, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if size is None and image is not None:
            size = image.size  # PIL.Image 自带 .size
        super().__init__(
            image=image,
            size=size,
            metadata=metadata or {},
            **kwargs,
        )

    @property
    def image(self) -> Any:
        """PIL 图像对象。"""
        return self._data["image"]

    @property
    def size(self) -> Optional[Tuple[int, int]]:
        """图像尺寸 ``(width, height)``。"""
        return self._data["size"]

    @property
    def metadata(self) -> Dict[str, Any]:
        """附加元数据。"""
        return self._data["metadata"]

    @classmethod
    def validate(cls, data: "MosaicData") -> bool:
        """校验图像数据：``image`` 应为 PIL.Image 或 None。"""
        if not isinstance(data, ImageData):
            return False
        img = data.get("image")
        if img is None:
            return True
        try:
            Image = _import_pil()
            return isinstance(img, Image.Image)
        except ImportError:
            return True  # 缺少 PIL 时跳过深度校验


# ---------------------------------------------------------------------------
# AudioData — 音频数据
# ---------------------------------------------------------------------------
class AudioData(MosaicData):
    """音频数据。

    Parameters
    ----------
    waveform:
        ``numpy.ndarray`` 波形数据，形状 ``(channels, samples)`` 或 ``(samples,)``。
    sample_rate:
        采样率（Hz）。
    metadata:
        附加元数据。
    """

    data_type = "audio"

    def __init__(
        self,
        waveform: Any = None,
        sample_rate: int = 22050,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            waveform=waveform,
            sample_rate=sample_rate,
            metadata=metadata or {},
            **kwargs,
        )

    @property
    def waveform(self) -> Any:
        """音频波形数组。"""
        return self._data["waveform"]

    @property
    def sample_rate(self) -> int:
        """采样率（Hz）。"""
        return self._data["sample_rate"]

    @property
    def metadata(self) -> Dict[str, Any]:
        """附加元数据。"""
        return self._data["metadata"]

    @classmethod
    def validate(cls, data: "MosaicData") -> bool:
        """校验音频数据：``sample_rate`` 必须为正整数。"""
        if not isinstance(data, AudioData):
            return False
        sr = data.get("sample_rate")
        return isinstance(sr, int) and sr > 0


# ---------------------------------------------------------------------------
# VideoData — 视频数据
# ---------------------------------------------------------------------------
class VideoData(MosaicData):
    """视频数据。

    Parameters
    ----------
    frames:
        ``PIL.Image`` 列表，每一帧一张图。
    fps:
        帧率。
    metadata:
        附加元数据。
    """

    data_type = "video"

    def __init__(
        self,
        frames: Optional[List[Any]] = None,
        fps: int = 30,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            frames=frames or [],
            fps=fps,
            metadata=metadata or {},
            **kwargs,
        )

    @property
    def frames(self) -> List[Any]:
        """视频帧列表。"""
        return self._data["frames"]

    @property
    def fps(self) -> int:
        """帧率。"""
        return self._data["fps"]

    @property
    def metadata(self) -> Dict[str, Any]:
        """附加元数据。"""
        return self._data["metadata"]

    @classmethod
    def validate(cls, data: "MosaicData") -> bool:
        """校验视频数据：``frames`` 必须为列表，``fps`` 为正数。"""
        if not isinstance(data, VideoData):
            return False
        if not isinstance(data.get("frames"), list):
            return False
        fps = data.get("fps")
        return isinstance(fps, (int, float)) and fps > 0


# ---------------------------------------------------------------------------
# SubtitleData — 字幕数据
# ---------------------------------------------------------------------------
class SubtitleData(MosaicData):
    """字幕数据。

    Parameters
    ----------
    segments:
        字幕片段列表，每个片段为 ``{"start": float, "end": float, "text": str}``。
    format:
        字幕格式，如 ``"srt"``、``"vtt"``、``"json"``。
    metadata:
        附加元数据。
    """

    data_type = "subtitle"

    _REQUIRED_KEYS = {"start", "end", "text"}

    def __init__(
        self,
        segments: Optional[List[Dict[str, Any]]] = None,
        format: str = "srt",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            segments=segments or [],
            format=format,
            metadata=metadata or {},
            **kwargs,
        )

    @property
    def segments(self) -> List[Dict[str, Any]]:
        """字幕片段列表。"""
        return self._data["segments"]

    @property
    def subtitle_format(self) -> str:
        """字幕格式。"""
        return self._data["format"]

    @property
    def metadata(self) -> Dict[str, Any]:
        """附加元数据。"""
        return self._data["metadata"]

    @classmethod
    def validate(cls, data: "MosaicData") -> bool:
        """校验字幕数据：每个片段须含 start/end/text 键。"""
        if not isinstance(data, SubtitleData):
            return False
        segments = data.get("segments")
        if not isinstance(segments, list):
            return False
        for seg in segments:
            if not isinstance(seg, dict):
                return False
            if not cls._REQUIRED_KEYS.issubset(seg.keys()):
                return False
        return True


# ---------------------------------------------------------------------------
# DocumentData — 文档数据
# ---------------------------------------------------------------------------
class DocumentData(MosaicData):
    """文档数据，主要用于 RAG 域。

    Parameters
    ----------
    chunks:
        文本分块列表。
    metadata:
        文档级元信息（文件名、页码、标题等）。
    chunk_metadata:
        每个 chunk 对应的元信息列表，长度与 ``chunks`` 一致。每个元素是
        一个字典，可包含 ``source``、``page``、``paragraph`` 等字段。
    """

    data_type = "document"

    def __init__(
        self,
        chunks: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            chunks=chunks or [],
            metadata=metadata or {},
            chunk_metadata=chunk_metadata or [],
            **kwargs,
        )

    @property
    def chunks(self) -> List[str]:
        """文本分块列表。"""
        return self._data["chunks"]

    @property
    def metadata(self) -> Dict[str, Any]:
        """文档级附加元数据。"""
        return self._data["metadata"]

    @property
    def chunk_metadata(self) -> List[Dict[str, Any]]:
        """每个 chunk 的元信息列表。"""
        return self._data["chunk_metadata"]

    @classmethod
    def validate(cls, data: "MosaicData") -> bool:
        """校验文档数据：``chunks`` 必须为字符串列表。"""
        if not isinstance(data, DocumentData):
            return False
        chunks = data.get("chunks")
        if not isinstance(chunks, list):
            return False
        return all(isinstance(c, str) for c in chunks)


# ---------------------------------------------------------------------------
# RagQueryResult — RAG 检索结果
# ---------------------------------------------------------------------------
class RagQueryResult(MosaicData):
    """RAG 检索结果数据。

    Parameters
    ----------
    query:
        原始查询文本。
    results:
        检索结果列表，每个 dict 包含 ``content`` (str)、``score`` (float)、
        ``source`` (str)、``metadata`` (dict)。
    answer:
        生成的回答文本（可选，由 CitationGenerator 填充）。
    citations:
        引用列表（可选），每个 dict 包含 ``citation_id``、``source``、
        ``content``、``score``。
    """

    data_type = "rag_query_result"

    def __init__(
        self,
        query: str = "",
        results: Optional[List[Dict[str, Any]]] = None,
        answer: Optional[str] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            query=query,
            results=results or [],
            answer=answer,
            citations=citations,
            **kwargs,
        )

    @property
    def query(self) -> str:
        """原始查询文本。"""
        return self._data["query"]

    @property
    def results(self) -> List[Dict[str, Any]]:
        """检索结果列表。"""
        return self._data["results"]

    @property
    def answer(self) -> Optional[str]:
        """生成的回答（可能为 None）。"""
        return self._data.get("answer")

    @property
    def citations(self) -> Optional[List[Dict[str, Any]]]:
        """引用列表（可能为 None）。"""
        return self._data.get("citations")

    @classmethod
    def validate(cls, data: "MosaicData") -> bool:
        """校验 RAG 检索结果。"""
        if not isinstance(data, RagQueryResult):
            return False
        results = data.get("results")
        if not isinstance(results, list):
            return False
        for item in results:
            if not isinstance(item, dict):
                return False
            if "content" not in item or "score" not in item:
                return False
        return True


# ---------------------------------------------------------------------------
# 类型注册表：支撑 from_dict 的多态分发
# ---------------------------------------------------------------------------
DATA_TYPE_REGISTRY: Dict[str, Type[MosaicData]] = {
    "mosaic": MosaicData,
    "text": TextData,
    "image": ImageData,
    "audio": AudioData,
    "video": VideoData,
    "subtitle": SubtitleData,
    "document": DocumentData,
    "rag_query_result": RagQueryResult,
}


def data_from_dict(data: Dict[str, Any]) -> MosaicData:
    """便捷函数：根据 ``__data_type__`` 字段自动分发到正确子类。"""
    return MosaicData.from_dict(data)
