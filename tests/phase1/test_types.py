# tests/phase1/test_types.py
"""Phase 1 数据类型测试。

覆盖 MosaicData 及所有子类的创建、序列化、反序列化与类型校验。
"""

from __future__ import annotations

import pytest

from mosaic.core.types import (
    MosaicData,
    TextData,
    ImageData,
    AudioData,
    VideoData,
    SubtitleData,
    DocumentData,
    DATA_TYPE_REGISTRY,
    data_from_dict,
)


# ===========================================================================
# T_TYPES_01: 创建 MosaicData，设置和获取值
# ===========================================================================
class TestMosaicDataBasic:
    """MosaicData 基本操作测试。"""

    def test_create_empty(self):
        """T_TYPES_01: 创建空的 MosaicData。"""
        md = MosaicData()
        assert len(md) == 0
        assert md.data_type == "mosaic"

    def test_create_with_kwargs(self):
        """T_TYPES_01: 通过关键字参数创建 MosaicData。"""
        md = MosaicData(key1="value1", key2=42, key3=True)
        assert len(md) == 3
        assert md["key1"] == "value1"
        assert md["key2"] == 42
        assert md["key3"] is True

    def test_dict_like_access(self):
        """T_TYPES_01: 字典式读写操作。"""
        md = MosaicData()
        md["a"] = 1
        md["b"] = "hello"
        assert md["a"] == 1
        assert md["b"] == "hello"
        assert "a" in md
        assert "c" not in md
        del md["a"]
        assert "a" not in md

    def test_get_with_default(self):
        """T_TYPES_01: get() 返回默认值。"""
        md = MosaicData(x=1)
        assert md.get("x") == 1
        assert md.get("missing") is None
        assert md.get("missing", "fallback") == "fallback"

    def test_update(self):
        """T_TYPES_01: update() 合并数据。"""
        md = MosaicData(a=1)
        md.update({"b": 2, "c": 3})
        assert md["b"] == 2
        assert md["c"] == 3

    def test_iteration(self):
        """T_TYPES_01: 迭代、keys、values、items。"""
        md = MosaicData(a=1, b=2)
        assert set(md) == {"a", "b"}
        assert list(md.keys()) == ["a", "b"]
        assert list(md.values()) == [1, 2]
        assert list(md.items()) == [("a", 1), ("b", 2)]

    def test_equality(self):
        """T_TYPES_01: 相等性比较。"""
        md1 = MosaicData(a=1, b=2)
        md2 = MosaicData(a=1, b=2)
        md3 = MosaicData(a=1, b=3)
        assert md1 == md2
        assert md1 != md3
        assert md1 == {"a": 1, "b": 2}

    def test_repr(self):
        """T_TYPES_01: repr 输出。"""
        md = MosaicData(x=1, y="hello")
        r = repr(md)
        assert "MosaicData" in r
        assert "x=1" in r


# ===========================================================================
# T_TYPES_02: TextData 的创建、序列化、反序列化
# ===========================================================================
class TestTextData:
    """TextData 测试。"""

    def test_create(self):
        """T_TYPES_02: 创建 TextData。"""
        td = TextData(content="Hello, World!", language="en")
        assert td.content == "Hello, World!"
        assert td.language == "en"
        assert td.data_type == "text"

    def test_defaults(self):
        """T_TYPES_02: 默认参数。"""
        td = TextData()
        assert td.content == ""
        assert td.language == "auto"

    def test_metadata(self):
        """T_TYPES_02: 元数据字段。"""
        td = TextData(content="Hi", metadata={"source": "test"})
        assert td.metadata == {"source": "test"}

    def test_serialization_roundtrip(self):
        """T_TYPES_02: 序列化后反序列化。"""
        td = TextData(content="Hello Mosaic", language="zh", metadata={"author": "test"})
        d = td.to_dict()
        restored = TextData.from_dict(d)
        assert isinstance(restored, TextData)
        assert restored.content == "Hello Mosaic"
        assert restored.language == "zh"
        assert restored.metadata == {"author": "test"}

    def test_data_from_dict(self):
        """T_TYPES_02: data_from_dict 正确分发到 TextData。"""
        td = TextData(content="test")
        d = td.to_dict()
        restored = data_from_dict(d)
        assert isinstance(restored, TextData)
        assert restored.content == "test"

    def test_validate(self):
        """T_TYPES_02: validate 校验。"""
        assert TextData.validate(TextData(content="hello"))
        assert TextData.validate(TextData(content=""))
        # 非 TextData 实例
        assert not TextData.validate(MosaicData(content="hello"))
        # content 不是字符串
        bad = TextData(content=123)  # type: ignore
        assert not TextData.validate(bad)


# ===========================================================================
# T_TYPES_03: ImageData 的创建、序列化、反序列化
# ===========================================================================
class TestImageData:
    """ImageData 测试。"""

    def test_create(self):
        """T_TYPES_03: 创建 ImageData。"""
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (100, 200), color="red")
        idata = ImageData(image=img)
        assert idata.size == (100, 200)
        assert idata.data_type == "image"

    def test_create_with_explicit_size(self):
        """T_TYPES_03: 显式指定尺寸。"""
        idata = ImageData(size=(640, 480))
        assert idata.size == (640, 480)
        assert idata.image is None

    def test_metadata(self):
        """T_TYPES_03: 元数据。"""
        idata = ImageData(metadata={"format": "png"})
        assert idata.metadata == {"format": "png"}

    def test_serialization_roundtrip(self):
        """T_TYPES_03: 序列化/反序列化还原 PIL 图像。"""
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (50, 50), color="blue")
        idata = ImageData(image=img, metadata={"desc": "blue square"})
        d = idata.to_dict()
        restored = ImageData.from_dict(d)
        assert isinstance(restored, ImageData)
        assert restored.size == (50, 50)
        assert restored.metadata == {"desc": "blue square"}
        # 还原后的图像应为 PIL Image
        from PIL import Image as PILImage2

        assert isinstance(restored.image, PILImage2.Image)
        assert restored.image.size == (50, 50)

    def test_validate(self):
        """T_TYPES_03: validate 校验。"""
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (10, 10))
        assert ImageData.validate(ImageData(image=img))
        assert ImageData.validate(ImageData(image=None))
        assert not ImageData.validate(MosaicData())


# ===========================================================================
# T_TYPES_04: AudioData 的创建、序列化、反序列化
# ===========================================================================
class TestAudioData:
    """AudioData 测试。"""

    def test_create(self):
        """T_TYPES_04: 创建 AudioData。"""
        import numpy as np

        wav = np.zeros((1, 16000), dtype=np.float32)
        adata = AudioData(waveform=wav, sample_rate=44100)
        assert adata.sample_rate == 44100
        assert adata.data_type == "audio"

    def test_defaults(self):
        """T_TYPES_04: 默认参数。"""
        adata = AudioData()
        assert adata.sample_rate == 22050
        assert adata.waveform is None

    def test_metadata(self):
        """T_TYPES_04: 元数据。"""
        adata = AudioData(metadata={"duration": 5.0})
        assert adata.metadata == {"duration": 5.0}

    def test_serialization_roundtrip(self):
        """T_TYPES_04: 序列化/反序列化还原 numpy 数组。"""
        import numpy as np

        wav = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        adata = AudioData(waveform=wav, sample_rate=16000, metadata={"ch": 1})
        d = adata.to_dict()
        restored = AudioData.from_dict(d)
        assert isinstance(restored, AudioData)
        assert restored.sample_rate == 16000
        assert restored.metadata == {"ch": 1}
        restored_wav = restored.waveform
        assert isinstance(restored_wav, np.ndarray)
        np.testing.assert_array_almost_equal(restored_wav, wav, decimal=5)

    def test_validate(self):
        """T_TYPES_04: validate 校验。"""
        assert AudioData.validate(AudioData(sample_rate=44100))
        assert not AudioData.validate(AudioData(sample_rate=0))
        assert not AudioData.validate(AudioData(sample_rate=-1))
        assert not AudioData.validate(MosaicData())


# ===========================================================================
# T_TYPES_05: VideoData 的创建、序列化、反序列化
# ===========================================================================
class TestVideoData:
    """VideoData 测试。"""

    def test_create(self):
        """T_TYPES_05: 创建 VideoData。"""
        vdata = VideoData(fps=60)
        assert vdata.fps == 60
        assert vdata.frames == []
        assert vdata.data_type == "video"

    def test_defaults(self):
        """T_TYPES_05: 默认参数。"""
        vdata = VideoData()
        assert vdata.fps == 30
        assert vdata.frames == []

    def test_metadata(self):
        """T_TYPES_05: 元数据。"""
        vdata = VideoData(metadata={"codec": "h264"})
        assert vdata.metadata == {"codec": "h264"}

    def test_serialization_roundtrip(self):
        """T_TYPES_05: 序列化/反序列化。"""
        from PIL import Image as PILImage

        frames = [
            PILImage.new("RGB", (32, 32), color="red"),
            PILImage.new("RGB", (32, 32), color="green"),
        ]
        vdata = VideoData(frames=frames, fps=24, metadata={"codec": "h264"})
        d = vdata.to_dict()
        restored = VideoData.from_dict(d)
        assert isinstance(restored, VideoData)
        assert restored.fps == 24
        assert restored.metadata == {"codec": "h264"}
        assert len(restored.frames) == 2
        from PIL import Image as PILImage2

        assert isinstance(restored.frames[0], PILImage2.Image)

    def test_validate(self):
        """T_TYPES_05: validate 校验。"""
        assert VideoData.validate(VideoData(frames=[], fps=30))
        assert not VideoData.validate(VideoData(fps=0))
        assert not VideoData.validate(VideoData(fps=-1))
        assert not VideoData.validate(MosaicData())


# ===========================================================================
# T_TYPES_06: SubtitleData 的创建和转换
# ===========================================================================
class TestSubtitleData:
    """SubtitleData 测试。"""

    def test_create(self):
        """T_TYPES_06: 创建 SubtitleData。"""
        segments = [
            {"start": 0.0, "end": 2.5, "text": "Hello"},
            {"start": 2.5, "end": 5.0, "text": "World"},
        ]
        sdata = SubtitleData(segments=segments, format="srt")
        assert sdata.format == "srt"
        assert len(sdata.segments) == 2
        assert sdata.data_type == "subtitle"

    def test_defaults(self):
        """T_TYPES_06: 默认参数。"""
        sdata = SubtitleData()
        assert sdata.format == "srt"
        assert sdata.segments == []

    def test_metadata(self):
        """T_TYPES_06: 元数据。"""
        sdata = SubtitleData(metadata={"lang": "zh"})
        assert sdata.metadata == {"lang": "zh"}

    def test_serialization_roundtrip(self):
        """T_TYPES_06: 序列化/反序列化。"""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Line 1"},
            {"start": 1.0, "end": 2.0, "text": "Line 2"},
        ]
        sdata = SubtitleData(segments=segments, format="vtt")
        d = sdata.to_dict()
        restored = SubtitleData.from_dict(d)
        assert isinstance(restored, SubtitleData)
        assert restored.format == "vtt"
        assert len(restored.segments) == 2
        assert restored.segments[0]["text"] == "Line 1"

    def test_validate(self):
        """T_TYPES_06: validate 校验。"""
        valid = SubtitleData(segments=[{"start": 0, "end": 1, "text": "ok"}])
        assert SubtitleData.validate(valid)
        # 缺少必要字段
        invalid = SubtitleData(segments=[{"start": 0, "end": 1}])
        assert not SubtitleData.validate(invalid)
        assert not SubtitleData.validate(MosaicData())


# ===========================================================================
# T_TYPES_07: DocumentData 的创建和转换
# ===========================================================================
class TestDocumentData:
    """DocumentData 测试。"""

    def test_create(self):
        """T_TYPES_07: 创建 DocumentData。"""
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        ddata = DocumentData(chunks=chunks)
        assert ddata.chunks == chunks
        assert ddata.data_type == "document"

    def test_defaults(self):
        """T_TYPES_07: 默认参数。"""
        ddata = DocumentData()
        assert ddata.chunks == []
        assert ddata.metadata == {}

    def test_metadata(self):
        """T_TYPES_07: 元数据。"""
        ddata = DocumentData(metadata={"source": "file.txt", "page": 1})
        assert ddata.metadata == {"source": "file.txt", "page": 1}

    def test_serialization_roundtrip(self):
        """T_TYPES_07: 序列化/反序列化。"""
        ddata = DocumentData(
            chunks=["Para 1", "Para 2"],
            metadata={"source": "test.pdf"},
        )
        d = ddata.to_dict()
        restored = DocumentData.from_dict(d)
        assert isinstance(restored, DocumentData)
        assert restored.chunks == ["Para 1", "Para 2"]
        assert restored.metadata == {"source": "test.pdf"}

    def test_validate(self):
        """T_TYPES_07: validate 校验。"""
        assert DocumentData.validate(DocumentData(chunks=["a", "b"]))
        assert DocumentData.validate(DocumentData(chunks=[]))
        # chunks 含非字符串
        assert not DocumentData.validate(DocumentData(chunks=[1, 2]))  # type: ignore
        assert not DocumentData.validate(MosaicData())


# ===========================================================================
# T_TYPES_08: 类型校验 —— 传入错误类型时抛出异常
# ===========================================================================
class TestTypeValidation:
    """类型校验 / 错误处理测试。"""

    def test_image_to_b64_rejects_non_image(self):
        """T_TYPES_08: _image_to_b64 拒绝非 PIL 图像。"""
        from mosaic.core.types import _image_to_b64

        with pytest.raises(TypeError, match="Expected PIL.Image.Image"):
            _image_to_b64("not an image")  # type: ignore

    def test_b64_to_image_rejects_invalid_token(self):
        """T_TYPES_08: _b64_to_image 拒绝非法 token。"""
        from mosaic.core.types import _b64_to_image

        with pytest.raises(ValueError, match="Invalid image token"):
            _b64_to_image("not:valid:token")

    def test_array_to_dict_rejects_non_array(self):
        """T_TYPES_08: _array_to_dict 拒绝非 numpy 数组。"""
        from mosaic.core.types import _array_to_dict

        with pytest.raises(TypeError, match="Expected numpy.ndarray"):
            _array_to_dict([1, 2, 3])  # type: ignore

    def test_tuple_preserved_in_roundtrip(self):
        """T_TYPES_08: 元组在序列化/反序列化中保留语义。"""
        md = MosaicData(coords=(10, 20), name="test")
        d = md.to_dict()
        restored = MosaicData.from_dict(d)
        assert isinstance(restored["coords"], tuple)
        assert restored["coords"] == (10, 20)

    def test_nested_tuple_roundtrip(self):
        """T_TYPES_08: 嵌套元组。"""
        md = MosaicData(data=((1, 2), (3, 4)))
        d = md.to_dict()
        restored = MosaicData.from_dict(d)
        assert restored["data"] == ((1, 2), (3, 4))