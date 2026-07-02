# mosaic/cli/doctor.py
"""环境诊断工具。

检查运行环境是否满足 Mosaic 框架的需求，包括：

* Python 版本
* 核心依赖（torch / transformers / diffusers）
* GPU 可用性与显存
* 可选依赖（imageio / soundfile / faiss / chromadb 等）
* 已注册节点数量
* 已加载插件数量
* 模型缓存目录

输出格式中：

- ``\u2713`` (✓) 表示通过
- ``\u26a0`` (⚠) 表示警告
- ``\u2717`` (✗) 表示错误

使用方式::

    python -m mosaic.cli.doctor
    # 或
    mosaic doctor
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
__all__ = ["run_doctor", "CheckResult"]

# ---------------------------------------------------------------------------
# 检查结果
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    """单项检查结果。

    Attributes
    ----------
    status:
        检查状态：``"ok"`` / ``"warn"`` / ``"error"``。
    message:
        检查结果描述。
    """

    status: str
    message: str


# 状态符号
_OK_SYMBOL = "\u2713"        # ✓
_WARN_SYMBOL = "\u26a0"      # ⚠
_ERROR_SYMBOL = "\u2717"     # ✗

_SYMBOLS = {
    "ok": _OK_SYMBOL,
    "warn": _WARN_SYMBOL,
    "error": _ERROR_SYMBOL,
}

# 可选依赖: (import 名, 显示名)
# 注意：insightface 和 onnxruntime 有专门的检查函数，不在此列表中
_OPTIONAL_PACKAGES: list[tuple[str, str]] = [
    # media 组
    ("imageio", "imageio"),
    ("imageio_ffmpeg", "imageio-ffmpeg"),
    ("soundfile", "soundfile"),
    ("librosa", "librosa"),
    ("edge_tts", "edge-tts"),
    ("trimesh", "trimesh"),
    ("skimage", "scikit-image"),
    # rag 组
    ("faiss", "faiss-cpu"),
    ("chromadb", "chromadb"),
    ("sentence_transformers", "sentence-transformers"),
    ("pdfplumber", "pdfplumber"),
    ("docx", "python-docx"),
    ("bs4", "beautifulsoup4"),
]


# ---------------------------------------------------------------------------
# 各项检查
# ---------------------------------------------------------------------------
def _check_python_version() -> CheckResult:
    """检查 Python 版本是否 >= 3.10。"""
    major, minor = sys.version_info[:2]
    patch = sys.version_info[2]
    version_str = f"{major}.{minor}.{patch}"
    if (major, minor) >= (3, 10):
        return CheckResult("ok", f"Python {version_str}")
    return CheckResult(
        "error",
        f"Python {version_str} — 需要 >= 3.10",
    )


def _check_package(
    import_name: str, display_name: str, required: bool
) -> CheckResult:
    """检查某个 Python 包是否已安装。

    Parameters
    ----------
    import_name:
        用于 ``importlib.import_module`` 的模块名。
    display_name:
        展示给用户的包名。
    required:
        是否为必需依赖；``True`` 时缺失记为错误，``False`` 时记为警告。
    """
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", None)
        if version:
            return CheckResult("ok", f"{display_name} 已安装 (v{version})")
        return CheckResult("ok", f"{display_name} 已安装")
    except ImportError:
        label = "必需依赖" if required else "可选依赖"
        status = "error" if required else "warn"
        return CheckResult(status, f"{display_name} 未安装（{label}）")
    except Exception as exc:  # noqa: BLE001
        return CheckResult("warn", f"{display_name} 导入失败: {exc}")


def _check_onnxruntime() -> CheckResult:
    """检查 onnxruntime 是否安装且 InferenceSession 可用。

    使用 ``mosaic.core.onnx_utils.is_onnxruntime_usable`` 进行深度检查，
    该函数不仅检查模块能否导入，还验证 ``InferenceSession`` 属性是否存在。
    """
    from mosaic.core.onnx_utils import OnnxRuntimeStatus

    usable, version, providers, error = OnnxRuntimeStatus.get()

    if not usable:
        # 根据错误信息生成修复建议
        if version and "InferenceSession" in (error or ""):
            return CheckResult(
                "warn",
                f"onnxruntime 已安装 (v{version})，但 InferenceSession 不可用\n"
                "    原因: C 扩展加载失败（CUDA/cuDNN 版本不匹配）\n"
                "    修复: pip install onnxruntime  # CPU 版本（兼容性最好）\n"
                "    或:   pip install onnxruntime-gpu==1.19.0  # PyTorch>=2.4 (cuDNN 9.x)\n"
                "    或:   pip install onnxruntime-gpu==1.17.1  # PyTorch<=2.3 (cuDNN 8.x)",
            )
        return CheckResult("warn", f"onnxruntime 不可用: {error}")

    has_gpu = providers and "CUDAExecutionProvider" in providers
    if has_gpu:
        return CheckResult(
            "ok",
            f"onnxruntime 已安装 (v{version}, GPU 加速可用)",
        )
    return CheckResult(
        "ok",
        f"onnxruntime 已安装 (v{version}, 仅 CPU)",
    )


def _check_insightface() -> CheckResult:
    """检查 insightface 是否安装且可正常导入。

    ``insightface`` 依赖 ``onnxruntime.InferenceSession``，当
    ``onnxruntime-gpu`` 的 C 扩展加载失败时，``insightface`` 会报
    ``AttributeError: module 'onnxruntime' has no attribute 'InferenceSession'``。
    本函数检测此问题并给出修复建议。
    """
    try:
        import insightface  # type: ignore # noqa: F401
    except ImportError:
        return CheckResult(
            "warn",
            "insightface 未安装（可选依赖，人脸检测和一致性域需要）",
        )
    except AttributeError as exc:
        # 最常见的情况：onnxruntime.InferenceSession 不可用
        if "InferenceSession" in str(exc) or "onnxruntime" in str(exc):
            return CheckResult(
                "warn",
                "insightface 导入失败: onnxruntime 的 InferenceSession 不可用\n"
                "    原因: onnxruntime-gpu 的 C 扩展加载失败（CUDA/cuDNN 版本不匹配）\n"
                "    修复: pip install onnxruntime-gpu==1.17.1  # 降级到兼容版本",
            )
        return CheckResult("warn", f"insightface 导入失败: {exc}")
    except Exception as exc:  # noqa: BLE001
        return CheckResult("warn", f"insightface 导入失败: {exc}")

    version = getattr(insightface, "__version__", None)
    if version:
        return CheckResult("ok", f"insightface 已安装 (v{version})")
    return CheckResult("ok", "insightface 已安装")


def _check_gpu() -> CheckResult:
    """检查 GPU 是否可用，返回名称与显存大小。

    依次检测 CUDA（NVIDIA）和 MPS（Apple Silicon）。
    """
    try:
        import torch  # type: ignore
    except ImportError:
        return CheckResult("warn", "GPU 检查跳过：torch 未安装")

    try:
        # 1. CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            return CheckResult(
                "ok",
                f"GPU 可用: {gpu_name} ({vram_gb:.1f} GB 显存, CUDA)",
            )

        # 2. MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return CheckResult(
                "ok",
                "GPU 可用: Apple Silicon (MPS)",
            )

        return CheckResult("warn", "GPU 不可用（将使用 CPU 推理）")
    except Exception as exc:  # noqa: BLE001
        return CheckResult("warn", f"GPU 检测失败: {exc}")


def _check_registered_nodes() -> CheckResult:
    """检查已注册节点数量。"""
    try:
        from mosaic.core.registry import registry

        registry.discover()
        count = len(registry)
        if count > 0:
            return CheckResult("ok", f"已注册 {count} 个节点")
        return CheckResult(
            "warn", "已注册 0 个节点（节点尚未被发现或注册）"
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("warn", f"节点注册表检查失败: {exc}")


def _check_plugins() -> CheckResult:
    """检查已加载插件数量。"""
    try:
        from mosaic.core.plugin import plugin_manager

        plugin_manager.load_plugins()
        count = len(plugin_manager)
        return CheckResult("ok", f"已加载 {count} 个插件")
    except Exception as exc:  # noqa: BLE001
        return CheckResult("warn", f"插件检查失败: {exc}")


def _check_model_cache() -> CheckResult:
    """检查模型缓存目录是否存在。

    依次检查 HuggingFace 相关的环境变量（``HF_HOME``、
    ``TRANSFORMERS_CACHE``、``HF_HUB_CACHE``、``HF_DATASETS_CACHE``）
    以及默认目录 ``~/.cache/huggingface``，只要其中任一目录存在即视为通过。
    """
    # 收集所有可能指向缓存目录的环境变量
    cache_env_vars = [
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
    ]
    found_dirs: list[str] = []
    for var in cache_env_vars:
        val = os.environ.get(var)
        if val and os.path.isdir(val):
            found_dirs.append(f"{var}={val}")

    # 跨平台构造默认缓存目录路径（避免硬编码 "/" 分隔符）。
    # 通过 MosaicEnv 集中读取 HF_HOME，保持与其它模块一致。
    from mosaic.core.env import MosaicEnv

    default_cache = MosaicEnv.get_hf_home()
    if os.path.isdir(default_cache):
        found_dirs.append(f"默认目录={default_cache}")

    if found_dirs:
        return CheckResult(
            "ok", f"模型缓存目录存在: {'; '.join(found_dirs)}"
        )

    # 所有缓存目录均不存在，报告优先级最高的目录（HF_HOME 或默认目录）
    primary = os.environ.get("HF_HOME") or default_cache
    return CheckResult(
        "warn",
        f"模型缓存目录不存在: {primary}（首次下载模型时将自动创建）",
    )


def _check_hf_auth() -> CheckResult:
    """检查 HuggingFace 认证状态。"""
    from mosaic.core.env import MosaicEnv

    token = MosaicEnv.get_hf_token()
    if token:
        # 验证 token 有效性（可选，避免网络请求）
        return CheckResult(
            "ok",
            "HuggingFace 认证已配置（HF_TOKEN 或 huggingface-cli login）",
        )
    return CheckResult(
        "warn",
        "未配置 HuggingFace 认证。下载 gated 模型（如 SVD、Llama）"
        "需要认证。设置方法：huggingface-cli login 或 "
        "export HF_TOKEN=your_token",
    )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def run_doctor() -> int:
    """运行环境诊断，返回退出码。

    依次执行所有检查项并打印结果，最后汇总警告与错误数量。
    存在 ``error`` 级别问题时返回 ``1``，否则返回 ``0``。

    Returns
    -------
    int
        退出码：``0`` 表示无错误，``1`` 表示存在 error 级别问题。
    """
    print()
    print("Mosaic 环境诊断")
    print("=" * 50)
    print()

    checks: list[tuple[str, CheckResult]] = []

    # Python 版本
    checks.append(("基础环境", _check_python_version()))

    # 核心依赖
    checks.append(("核心依赖", _check_package("torch", "torch", required=True)))
    checks.append(("核心依赖", _check_package("transformers", "transformers", required=True)))
    checks.append(("核心依赖", _check_package("diffusers", "diffusers", required=True)))

    # GPU
    checks.append(("GPU", _check_gpu()))

    # 可选依赖（简单检查）
    for import_name, display_name in _OPTIONAL_PACKAGES:
        checks.append(("可选依赖", _check_package(import_name, display_name, required=False)))

    # insightface 和 onnxruntime 需要深度检查（验证关键 API 可用性）
    checks.append(("可选依赖", _check_insightface()))
    checks.append(("可选依赖", _check_onnxruntime()))

    # 节点与插件
    checks.append(("框架状态", _check_registered_nodes()))
    checks.append(("框架状态", _check_plugins()))

    # 模型缓存目录
    checks.append(("模型缓存", _check_model_cache()))

    # HuggingFace 认证
    checks.append(("HuggingFace", _check_hf_auth()))

    # 输出结果（按分组打印）
    warn_count = 0
    error_count = 0
    current_group = ""
    for group, result in checks:
        if group != current_group:
            current_group = group
            print(f"\n  [{group}]")
        symbol = _SYMBOLS.get(result.status, "?")
        # 多行消息：首行带符号，后续行缩进对齐
        lines = result.message.split("\n")
        print(f"  {symbol}  {lines[0]}")
        for line in lines[1:]:
            print(f"     {line}")
        if result.status == "warn":
            warn_count += 1
        elif result.status == "error":
            error_count += 1

    print()
    print(f"诊断完成: {warn_count} 个警告, {error_count} 个错误")
    print()

    # 存在 error 级别问题时返回非零退出码，便于脚本/CI 据此判断环境是否就绪
    return 1 if error_count > 0 else 0


# ---------------------------------------------------------------------------
# 直接运行入口
# ---------------------------------------------------------------------------
def main() -> int:
    """命令行入口：运行环境诊断。

    Returns
    -------
    int
        进程退出码，``0`` 表示无错误，``1`` 表示存在错误。
    """
    try:
        return run_doctor()
    except Exception as exc:  # noqa: BLE001
        print(f"诊断过程中发生错误: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
