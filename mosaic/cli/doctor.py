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
_OPTIONAL_PACKAGES: list[tuple[str, str]] = [
    ("imageio", "imageio"),
    ("imageio_ffmpeg", "imageio-ffmpeg"),
    ("soundfile", "soundfile"),
    ("librosa", "librosa"),
    ("faiss", "faiss-cpu"),
    ("chromadb", "chromadb"),
    ("sentence_transformers", "sentence-transformers"),
    ("insightface", "insightface"),
    ("onnxruntime", "onnxruntime"),
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


def _check_gpu() -> CheckResult:
    """检查 GPU 是否可用，返回名称与显存大小。"""
    try:
        import torch  # type: ignore
    except ImportError:
        return CheckResult("warn", "GPU 检查跳过：torch 未安装")

    try:
        if not torch.cuda.is_available():
            return CheckResult("warn", "GPU 不可用（将使用 CPU 推理）")
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        return CheckResult(
            "ok",
            f"GPU 可用: {gpu_name} ({vram_gb:.1f} GB 显存)",
        )
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
    """检查模型缓存目录是否存在。"""
    cache_dir = os.environ.get(
        "HF_HOME",
        os.path.expanduser("~/.cache/huggingface"),
    )
    if os.path.isdir(cache_dir):
        return CheckResult("ok", f"模型缓存目录存在: {cache_dir}")
    return CheckResult(
        "warn",
        f"模型缓存目录不存在: {cache_dir}（首次下载模型时将自动创建）",
    )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def run_doctor() -> int:
    """运行环境诊断，返回警告数量。

    依次执行所有检查项并打印结果，最后汇总警告与错误数量。

    Returns
    -------
    int
        警告数量。
    """
    print()
    print("Mosaic 环境诊断")
    print("=" * 50)
    print()

    checks: list[CheckResult] = []

    # Python 版本
    checks.append(_check_python_version())

    # 核心依赖
    checks.append(_check_package("torch", "torch", required=True))
    checks.append(_check_package("transformers", "transformers", required=True))
    checks.append(_check_package("diffusers", "diffusers", required=True))

    # GPU
    checks.append(_check_gpu())

    # 可选依赖
    for import_name, display_name in _OPTIONAL_PACKAGES:
        checks.append(_check_package(import_name, display_name, required=False))

    # 节点与插件
    checks.append(_check_registered_nodes())
    checks.append(_check_plugins())

    # 模型缓存目录
    checks.append(_check_model_cache())

    # 输出结果
    warn_count = 0
    error_count = 0
    for result in checks:
        symbol = _SYMBOLS.get(result.status, "?")
        print(f"  {symbol}  {result.message}")
        if result.status == "warn":
            warn_count += 1
        elif result.status == "error":
            error_count += 1

    print()
    print(f"诊断完成: {warn_count} 个警告, {error_count} 个错误")
    print()

    return warn_count


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
        run_doctor()
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"诊断过程中发生错误: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
