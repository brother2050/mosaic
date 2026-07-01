"""ONNX Runtime 兼容性工具模块。

解决 ``onnxruntime-gpu`` 在 CUDA/cuDNN 版本不匹配时，
模块可导入但 ``InferenceSession`` 不可用的问题。

核心功能：
- :func:`is_onnxruntime_usable` — 检查 onnxruntime 是否真正可用
- :func:`create_inference_session` — 安全创建 InferenceSession，含 preload_dlls 和 fallback
- :func:`get_onnx_providers` — 获取可用的 Execution Provider 列表

使用方式::

    from mosaic.core.onnx_utils import is_onnxruntime_usable, create_inference_session

    if is_onnxruntime_usable():
        session = create_inference_session("model.onnx", providers=["CUDAExecutionProvider"])
    else:
        # 回退到其他推理后端
        ...
"""

from __future__ import annotations

import logging
from typing import Any

__all__ = [
    "is_onnxruntime_usable",
    "create_inference_session",
    "get_onnx_providers",
    "OnnxRuntimeStatus",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 状态缓存（避免重复检查）
# ---------------------------------------------------------------------------
class OnnxRuntimeStatus:
    """onnxruntime 可用性检查结果（单例缓存）。"""

    _checked: bool = False
    _usable: bool = False
    _version: str | None = None
    _providers: list[str] | None = None
    _error: str | None = None

    @classmethod
    def reset(cls) -> None:
        """重置缓存（用于测试）。"""
        cls._checked = False
        cls._usable = False
        cls._version = None
        cls._providers = None
        cls._error = None

    @classmethod
    def get(
        cls,
    ) -> tuple[bool, str | None, list[str] | None, str | None]:
        """获取检查结果，首次调用时执行检查。

        Returns
        -------
        tuple
            (usable, version, providers, error_message)
        """
        if not cls._checked:
            cls._check()
        return cls._usable, cls._version, cls._providers, cls._error

    @classmethod
    def _check(cls) -> None:
        """执行实际的 onnxruntime 可用性检查。"""
        cls._checked = True

        try:
            import onnxruntime as ort  # type: ignore
        except ImportError:
            cls._error = "onnxruntime 未安装"
            return
        except Exception as exc:  # noqa: BLE001
            cls._error = f"onnxruntime 导入失败: {exc}"
            return

        cls._version = getattr(ort, "__version__", "未知")

        # 关键检查：InferenceSession 是否存在
        # onnxruntime-gpu 在 CUDA/cuDNN 不匹配时，C 扩展加载失败，
        # 模块可导入但 InferenceSession 属性不存在
        if not hasattr(ort, "InferenceSession"):
            cls._error = (
                f"onnxruntime 已安装 (v{cls._version})，但 InferenceSession 不可用"
                f"（C 扩展加载失败，通常是 CUDA/cuDNN 版本不匹配）"
            )
            return

        # 检查可用的 Execution Provider
        try:
            cls._providers = list(ort.get_available_providers())
        except Exception as exc:  # noqa: BLE001
            cls._error = f"获取 providers 失败: {exc}"
            return

        cls._usable = True


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------
def is_onnxruntime_usable() -> bool:
    """检查 onnxruntime 是否真正可用。

    不仅检查模块能否 ``import``，还验证 ``InferenceSession`` 属性是否存在。
    ``onnxruntime-gpu`` 在 CUDA/cuDNN 版本不匹配时，模块可以导入，
    但 ``InferenceSession`` 不存在（C 扩展加载失败）。

    Returns
    -------
    bool
        ``True`` 表示 onnxruntime 可正常使用。
    """
    usable, _, _, _ = OnnxRuntimeStatus.get()
    return usable


def get_onnx_providers(device: str = "cuda") -> list[str]:
    """获取可用的 Execution Provider 列表，按优先级排序。

    Parameters
    ----------
    device:
        目标设备，``"cuda"``、``"mps"`` 或 ``"cpu"``。

    Returns
    -------
    list[str]
        Provider 列表，如 ``["CUDAExecutionProvider", "CPUExecutionProvider"]``、
        ``["CoreMLExecutionProvider", "CPUExecutionProvider"]``。
        如果 onnxruntime 不可用，返回空列表。
    """
    usable, _, providers, _ = OnnxRuntimeStatus.get()
    if not usable or providers is None:
        return []

    if device.startswith("cuda") and "CUDAExecutionProvider" in providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device.startswith("mps") and "CoreMLExecutionProvider" in providers:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def create_inference_session(
    model_path: str,
    providers: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """安全创建 ONNX Runtime InferenceSession。

    在创建会话前，自动调用 ``preload_dlls()``（onnxruntime-gpu >= 1.21.0）
    以加载 PyTorch 安装的 CUDA/cuDNN 库，避免版本不匹配问题。

    Parameters
    ----------
    model_path:
        ONNX 模型文件路径。
    providers:
        Execution Provider 列表。如为 ``None``，自动检测。
    **kwargs:
        传递给 ``InferenceSession`` 的额外参数。

    Returns
    -------
    onnxruntime.InferenceSession
        创建的推理会话。

    Raises
    ------
    RuntimeError
        如果 onnxruntime 不可用或创建会话失败。
    """
    usable, version, available_providers, error = OnnxRuntimeStatus.get()
    if not usable:
        raise RuntimeError(
            f"onnxruntime 不可用: {error}\n"
            "修复建议:\n"
            "  1. pip install onnxruntime  # CPU 版本\n"
            "  2. 或 pip install onnxruntime-gpu  # GPU 版本（需匹配 CUDA/cuDNN）\n"
            "  3. PyTorch >= 2.4 (cuDNN 9.x) 用户请安装 onnxruntime-gpu>=1.19.0\n"
            "  4. PyTorch <= 2.3 (cuDNN 8.x) 用户请安装 onnxruntime-gpu==1.17.1"
        )

    import onnxruntime as ort  # type: ignore

    # onnxruntime-gpu >= 1.21.0 提供 preload_dlls，可自动加载 PyTorch 的 CUDA/cuDNN 库
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
            logger.debug("onnxruntime preload_dlls() 成功")
        except Exception as exc:  # noqa: BLE001
            logger.debug("onnxruntime preload_dlls() 失败（可忽略）: %s", exc)

    # 自动选择 providers
    if providers is None:
        if available_providers and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

    try:
        session = ort.InferenceSession(model_path, providers=providers, **kwargs)
        actual_providers = session.get_providers()
        logger.info(
            "ONNX InferenceSession 创建成功 (providers=%s, ort_version=%s)",
            actual_providers,
            version,
        )
        return session
    except Exception as exc:  # noqa: BLE001
        # 如果 CUDA provider 失败，尝试回退到 CPU
        if "CUDAExecutionProvider" in providers:
            logger.warning(
                "CUDA ExecutionProvider 创建失败: %s，回退到 CPU", exc
            )
            try:
                session = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"], **kwargs
                )
                logger.info("ONNX InferenceSession 创建成功 (CPU 回退)")
                return session
            except Exception as cpu_exc:  # noqa: BLE001
                raise RuntimeError(
                    f"创建 InferenceSession 失败 (CUDA 和 CPU 均不可用):\n"
                    f"  CUDA 错误: {exc}\n"
                    f"  CPU 错误: {cpu_exc}"
                ) from cpu_exc
        raise RuntimeError(f"创建 InferenceSession 失败: {exc}") from exc
