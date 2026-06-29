# mosaic/nodes/audio/tts_backends/hf_model_manager.py
"""HuggingFace 模型管理器。

提供统一的 TTS 模型下载与路径解析能力：

* **自动下载**：本地路径不存在时，从 HuggingFace（或镜像）下载模型。
* **路径解析**：各后端在 HF 仓库中的文件布局各不相同，提供统一的
  :meth:`HFModelManager.find_file` / :meth:`HFModelManager.find_dir`
  方法按候选列表查找。
* **镜像支持**：通过 ``HF_ENDPOINT`` 或 ``MOSAIC_HF_MIRROR`` 环境变量
  配置镜像地址，默认使用 ``https://hf-mirror.com``。

各后端默认 HF 仓库
------------------

============== ==============================================
后端           HF Repo ID
============== ==============================================
ChatTTS        ``2Noise/ChatTTS``
Fish Speech    ``fishaudio/fish-speech-1.5``
GPT-SoVITS     ``lj1995/GPT-SoVITS``
CosyVoice      ``FunAudioLLM/CosyVoice2-0.5B``
============== ==============================================

使用示例
--------

.. code-block:: python

    from mosaic.nodes.audio.tts_backends.hf_model_manager import HFModelManager

    # 确保模型已下载（本地存在则跳过）
    model_dir = HFModelManager.ensure_model(
        model_path="/data/chattts",
        repo_id="2Noise/ChatTTS",
    )

    # 按候选列表查找文件
    dvae_path = HFModelManager.find_file(model_dir, [
        "asset/DVAE.safetensors",
        "asset/DVAE.pt",
        "asset/Decoder.safetensors",
        "dvae.safetensors",
    ])

设计要点
--------
* ``huggingface_hub`` 为可选依赖，未安装时回退到 ``git clone``。
* ``GIT_LFS_SKIP_SMUDGE=1`` 用于跳过大文件下载（仅克隆仓库结构），
  随后按需下载具体文件。
* 所有方法均为 ``@staticmethod``，无需实例化。
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any

__all__ = ["HFModelManager"]

logger = logging.getLogger("mosaic.tts.backends.hf_model_manager")


class HFModelManager:
    """HuggingFace 模型管理器（静态工具类）。

    提供模型下载与路径解析能力，所有方法均为静态方法。
    """

    # 各后端默认 HF 仓库 ID
    DEFAULT_REPOS: dict[str, str] = {
        "chattts": "2Noise/ChatTTS",
        "fish": "fishaudio/fish-speech-1.5",
        "sovits": "lj1995/GPT-SoVITS",
        "cosyvoice": "FunAudioLLM/CosyVoice2-0.5B",
    }

    @staticmethod
    def get_mirror() -> str:
        """获取 HF 镜像地址。

        优先级：``HF_ENDPOINT`` > ``MOSAIC_HF_MIRROR`` > 默认镜像。

        Returns
        -------
        str
            镜像基础 URL（不含尾部斜杠）。
        """
        mirror = (
            os.environ.get("HF_ENDPOINT")
            or os.environ.get("MOSAIC_HF_MIRROR")
            or "https://hf-mirror.com"
        )
        return mirror.rstrip("/")

    @staticmethod
    def ensure_model(
        model_path: str,
        repo_id: str | None = None,
        backend_name: str | None = None,
    ) -> str:
        """确保模型存在本地，不存在则从 HF 下载。

        Parameters
        ----------
        model_path : str
            本地模型目录路径。如果目录存在且非空，直接返回。
        repo_id : str | None
            HF 仓库 ID（如 ``"2Noise/ChatTTS"``）。
            如果为 ``None`` 且 ``backend_name`` 提供，则使用默认仓库。
        backend_name : str | None
            后端名称（``"chattts"`` / ``"fish"`` / ``"sovits"`` /
            ``"cosyvoice"``），用于查找默认仓库 ID。

        Returns
        -------
        str
            本地模型目录路径。

        Raises
        ------
        FileNotFoundError
            本地路径不存在且无法下载（无 repo_id 或下载失败）。
        """
        # 本地路径存在且非空 → 直接返回
        if os.path.isdir(model_path) and os.listdir(model_path):
            logger.debug("Model found locally at %s.", model_path)
            return model_path

        # 解析 repo_id
        if repo_id is None and backend_name is not None:
            repo_id = HFModelManager.DEFAULT_REPOS.get(backend_name)
        if repo_id is None:
            raise FileNotFoundError(
                f"Model path '{model_path}' does not exist and no "
                f"repo_id or backend_name provided for download."
            )

        # 创建父目录
        parent_dir = os.path.dirname(os.path.abspath(model_path))
        os.makedirs(parent_dir, exist_ok=True)

        # 下载
        logger.info(
            "Model not found at %s. Downloading from HF repo %r ...",
            model_path,
            repo_id,
        )
        return HFModelManager._download(repo_id, model_path)

    @staticmethod
    def _download(repo_id: str, local_dir: str) -> str:
        """从 HuggingFace 下载模型仓库。

        下载策略：
        1. 优先使用 ``huggingface_hub.snapshot_download``（如果安装）。
        2. 回退到 ``git clone``（使用镜像 URL）。

        Parameters
        ----------
        repo_id : str
            HF 仓库 ID（如 ``"2Noise/ChatTTS"``）。
        local_dir : str
            本地目标目录。

        Returns
        -------
        str
            下载后的本地目录路径。

        Raises
        ------
        RuntimeError
            下载失败。
        """
        # 策略 1: huggingface_hub
        try:
            from huggingface_hub import snapshot_download  # type: ignore

            mirror = HFModelManager.get_mirror()
            os.environ.setdefault("HF_ENDPOINT", mirror)

            logger.info(
                "Using huggingface_hub to download %r to %s ...",
                repo_id,
                local_dir,
            )
            result = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                # 不设置 allow_patterns，下载全部文件
            )
            if isinstance(result, str):
                return result
            return local_dir
        except ImportError:
            logger.debug(
                "huggingface_hub not installed, falling back to git clone."
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "huggingface_hub download failed: %s. "
                "Falling back to git clone.",
                exc,
            )

        # 策略 2: git clone
        mirror = HFModelManager.get_mirror()
        clone_url = f"{mirror}/{repo_id}"

        logger.info("Cloning %s to %s ...", clone_url, local_dir)

        env = os.environ.copy()
        # 跳过 LFS 大文件下载（用户可后续按需拉取）
        env["GIT_LFS_SKIP_SMUDGE"] = "1"

        result = subprocess.run(
            ["git", "clone", clone_url, local_dir],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to clone {clone_url}: {result.stderr.strip()}"
            )

        logger.info("Successfully cloned %r to %s.", repo_id, local_dir)
        return local_dir

    @staticmethod
    def find_file(
        directory: str,
        candidates: list[str] | tuple[str, ...],
    ) -> str:
        """在目录中按候选列表查找文件。

        Parameters
        ----------
        directory : str
            搜索根目录。
        candidates : list[str] | tuple[str, ...]
            候选相对路径列表（按优先级排序）。

        Returns
        -------
        str
            找到的文件绝对路径；未找到返回空字符串 ``""``。
        """
        for candidate in candidates:
            path = os.path.join(directory, candidate)
            if os.path.isfile(path):
                return path
        return ""

    @staticmethod
    def find_dir(
        directory: str,
        candidates: list[str] | tuple[str, ...],
    ) -> str:
        """在目录中按候选列表查找子目录。

        Parameters
        ----------
        directory : str
            搜索根目录。
        candidates : list[str] | tuple[str, ...]
            候选相对路径列表（按优先级排序）。

        Returns
        -------
        str
            找到的目录绝对路径；未找到返回空字符串 ``""``。
        """
        for candidate in candidates:
            path = os.path.join(directory, candidate)
            if os.path.isdir(path):
                return path
        return ""

    @staticmethod
    def find_weight(
        directory: str,
        basename: str,
        extensions: list[str] | None = None,
        subdirs: list[str] | None = None,
    ) -> str:
        """按文件名和扩展名候选查找权重文件。

        在 ``directory`` 及可选的子目录中查找 ``basename`` 加上各种扩展名
        的文件。例如 ``find_weight(dir, "DVAE")`` 会查找
        ``DVAE.safetensors`` / ``DVAE.pt`` / ``DVAE.pth`` / ``DVAE.bin``。

        Parameters
        ----------
        directory : str
            搜索根目录。
        basename : str
            文件基本名（不含扩展名）。
        extensions : list[str] | None
            扩展名候选列表（含点号），默认
            ``[".safetensors", ".pt", ".pth", ".bin", ".ckpt"]``。
        subdirs : list[str] | None
            额外搜索的子目录列表（如 ``["asset", ""]``）。

        Returns
        -------
        str
            找到的文件绝对路径；未找到返回空字符串。
        """
        if extensions is None:
            extensions = [".safetensors", ".pt", ".pth", ".bin", ".ckpt"]

        search_dirs = [directory]
        if subdirs:
            for sd in subdirs:
                search_dirs.append(os.path.join(directory, sd))

        for search_dir in search_dirs:
            for ext in extensions:
                path = os.path.join(search_dir, basename + ext)
                if os.path.isfile(path):
                    return path
        return ""

    @staticmethod
    def load_yaml_config(config_path: str) -> dict[str, Any]:
        """加载 YAML 配置文件。

        Parameters
        ----------
        config_path : str
            YAML 文件路径。

        Returns
        -------
        dict[str, Any]
            配置字典；文件不存在或解析失败返回空字典。
        """
        if not os.path.isfile(config_path):
            return {}
        try:
            import yaml  # type: ignore

            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
        except ImportError:
            logger.debug("PyYAML not installed, cannot load %s.", config_path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to load YAML %s: %s", config_path, exc)
        return {}

    @staticmethod
    def load_json_config(config_path: str) -> dict[str, Any]:
        """加载 JSON 配置文件。

        Parameters
        ----------
        config_path : str
            JSON 文件路径。

        Returns
        -------
        dict[str, Any]
            配置字典；文件不存在或解析失败返回空字典。
        """
        if not os.path.isfile(config_path):
            return {}
        try:
            import json

            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to load JSON %s: %s", config_path, exc)
        return {}
