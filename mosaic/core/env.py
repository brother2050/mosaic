# mosaic/core/env.py
"""集中管理 Mosaic 运行期环境变量。

历史上各模块（scheduler、doctor、hf_model_manager 等）分散读取
``os.environ``，存在重复逻辑与不一致的默认值。本模块提供统一的
:class:`MosaicEnv` 入口，集中读取并归一化 Mosaic 相关环境变量。

设计要点
--------
* 全部方法为 ``@staticmethod``，无需实例化即可使用。
* 读取失败或变量未设置时返回安全的默认值（``None`` 或语义化默认），
  不抛异常，避免影响启动流程。
* 仅做“读取 + 归一化”，不缓存，便于运行期通过 ``os.environ`` 热更新。

常用变量
--------
================================  ============================================
变量名                            含义
================================  ============================================
``MOSAIC_MEMORY_LIMIT``           调度器显存上限（GB）
``MOSAIC_LOW_MEMORY``             是否启用低显存模式（``1``/``true``）
``CUDA_VISIBLE_DEVICES``          可见 CUDA 设备
``HF_HOME``                       HuggingFace 缓存主目录
``HF_ENDPOINT``                   HuggingFace 镜像端点
``MOSAIC_HF_MIRROR``              Mosaic 专用 HF 镜像（备选）
``MOSAIC_LOG_LEVEL``              Mosaic 日志级别
================================  ============================================
"""
from __future__ import annotations

import os

__all__ = ["MosaicEnv"]

#: 布尔意义上的“真”值字符串（大小写不敏感）。
_TRUE_VALUES = {"1", "true", "yes", "on", "y", "t"}


class MosaicEnv:
    """Mosaic 环境变量的集中读取入口。

    所有方法均为静态方法，直接读取 ``os.environ``，不缓存结果。
    """

    # ------------------------------------------------------------------
    # 计算资源相关
    # ------------------------------------------------------------------
    @staticmethod
    def get_memory_limit() -> float | None:
        """返回调度器显存上限（GB）。

        读取 ``MOSAIC_MEMORY_LIMIT`` 环境变量。未设置或无法解析时返回
        ``None``，表示由调度器使用 GPU 实际总量。
        """
        raw = os.environ.get("MOSAIC_MEMORY_LIMIT")
        if raw is None or raw.strip() == "":
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def is_low_memory_mode() -> bool:
        """是否启用低显存模式（``MOSAIC_LOW_MEMORY``）。

        取值为 ``1``/``true``/``yes``/``on`` 时视为启用。
        """
        raw = os.environ.get("MOSAIC_LOW_MEMORY", "")
        return raw.strip().lower() in _TRUE_VALUES

    @staticmethod
    def get_cuda_visible_devices() -> str | None:
        """返回 ``CUDA_VISIBLE_DEVICES`` 的原始值。

        未设置时返回 ``None``（表示不限制可见设备）。
        """
        raw = os.environ.get("CUDA_VISIBLE_DEVICES")
        if raw is None or raw.strip() == "":
            return None
        return raw.strip()

    @staticmethod
    def get_device() -> str | None:
        """返回强制指定的计算设备（``MOSAIC_DEVICE``）。

        例如 ``"cuda"``、``"cpu"``、``"cuda:1"``。未设置时返回 ``None``，
        由调度器自动检测。
        """
        raw = os.environ.get("MOSAIC_DEVICE")
        if raw is None or raw.strip() == "":
            return None
        return raw.strip()

    # ------------------------------------------------------------------
    # HuggingFace 相关
    # ------------------------------------------------------------------
    @staticmethod
    def get_hf_home() -> str:
        """返回 HuggingFace 缓存主目录。

        优先级：``HF_HOME`` 环境变量 > 默认目录 ``~/.cache/huggingface``。
        跨平台构造路径，避免硬编码 ``/`` 分隔符。
        """
        raw = os.environ.get("HF_HOME")
        if raw and raw.strip():
            return raw.strip()
        return os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

    @staticmethod
    def get_hf_endpoint() -> str | None:
        """返回 HuggingFace 镜像端点（``HF_ENDPOINT``）。

        未设置时返回 ``None``，由调用方决定是否使用 ``MOSAIC_HF_MIRROR``
        或默认镜像。
        """
        raw = os.environ.get("HF_ENDPOINT")
        if raw is None or raw.strip() == "":
            return None
        return raw.strip().rstrip("/")

    @staticmethod
    def get_hf_mirror() -> str | None:
        """返回 Mosaic 专用 HF 镜像（``MOSAIC_HF_MIRROR``）。

        未设置时返回 ``None``。
        """
        raw = os.environ.get("MOSAIC_HF_MIRROR")
        if raw is None or raw.strip() == "":
            return None
        return raw.strip().rstrip("/")

    @staticmethod
    def get_hf_endpoint_or_mirror(default: str | None = None) -> str | None:
        """返回 HF 下载端点，按优先级合并镜像配置。

        优先级：``HF_ENDPOINT`` > ``MOSAIC_HF_MIRROR`` > *default*。
        """
        return (
            MosaicEnv.get_hf_endpoint()
            or MosaicEnv.get_hf_mirror()
            or (default.rstrip("/") if default else None)
        )

    # ------------------------------------------------------------------
    # 日志相关
    # ------------------------------------------------------------------
    @staticmethod
    def get_log_level() -> str | None:
        """返回 Mosaic 日志级别（``MOSAIC_LOG_LEVEL``）。

        如 ``"DEBUG"``、``"INFO"``。未设置时返回 ``None``。
        """
        raw = os.environ.get("MOSAIC_LOG_LEVEL")
        if raw is None or raw.strip() == "":
            return None
        return raw.strip().upper()
