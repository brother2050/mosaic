# mosaic/nodes/audio/tts_backends/weights/converter.py
"""TTS 权重转换框架。

定义权重转换器抽象基类 :class:`WeightConverter`，将各 TTS 后端的原始
checkpoint 转换为 Mosaic 标准权重格式（safetensors + ``config.json``）。
每个后端实现自己的子类完成具体的转换逻辑。

Mosaic 标准权重格式
------------------
* 使用 safetensors 格式存储权重；
* 每个组件独立一个文件：``text_frontend.safetensors``、
  ``acoustic_model.safetensors``、``vocoder.safetensors``；
* 配置文件：``config.json``（包含模型配置、版本信息、原始来源）。

依赖说明
--------
``torch`` 与 ``safetensors`` 均为惰性导入，仅在保存权重时需要；模块加载
本身不依赖它们。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["WeightConverter"]


class WeightConverter(ABC):
    """TTS 权重转换器抽象基类。

    每个 TTS 后端实现自己的转换器，将原始 checkpoint 转换为
    Mosaic 标准权重格式。

    Mosaic 标准权重格式：
    - 使用 safetensors 格式存储权重
    - 每个组件独立一个文件：
      text_frontend.safetensors、acoustic_model.safetensors、vocoder.safetensors
    - 配置文件：config.json（包含模型配置、版本信息、原始来源）
    """

    # 标准组件名称
    COMPONENTS = ("text_frontend", "acoustic_model", "vocoder")

    @abstractmethod
    def convert(
        self,
        source_path: str,
        output_path: str,
        components: list[str] | None = None,
    ) -> dict[str, str]:
        """将原始 checkpoint 转换为 Mosaic 标准格式。

        Parameters
        ----------
        source_path : str
            原始权重文件路径或目录。
        output_path : str
            输出目录路径。
        components : list[str] | None
            要转换的组件列表，None 表示全部。
            可选值：["text_frontend", "acoustic_model", "vocoder"]

        Returns
        -------
        dict[str, str]
            {组件名: 输出文件路径} 映射。

        Raises
        ------
        FileNotFoundError
            源路径不存在。
        ValueError
            源格式不支持或组件名称无效。
        """

    @abstractmethod
    def validate(self, converted_path: str) -> bool:
        """验证转换后的权重是否完整可用。

        Parameters
        ----------
        converted_path : str
            转换后的权重目录路径。

        Returns
        -------
        bool
            True 表示验证通过。
        """

    @staticmethod
    def list_formats(source_path: str) -> list[str]:
        """检测源路径中的权重格式。

        支持检测的格式：
        - "safetensors"：.safetensors 文件
        - "pytorch"：.pt / .pth / .bin 文件
        - "checkpoint"：包含 .index.json 的检查点目录
        - "safetensors_dir"：包含 .safetensors 文件的目录

        Parameters
        ----------
        source_path : str
            源路径（文件或目录）。

        Returns
        -------
        list[str]
            检测到的格式列表。
        """
        import os

        formats: list[str] = []
        if os.path.isdir(source_path):
            files = os.listdir(source_path)
            if any(f.endswith(".safetensors") for f in files):
                formats.append("safetensors_dir")
            if any(f.endswith((".pt", ".pth", ".bin")) for f in files):
                formats.append("pytorch")
            if any(f.endswith(".index.json") for f in files):
                formats.append("checkpoint")
        elif os.path.isfile(source_path):
            if source_path.endswith(".safetensors"):
                formats.append("safetensors")
            elif source_path.endswith((".pt", ".pth", ".bin")):
                formats.append("pytorch")

        return formats

    @staticmethod
    def _save_config(config: dict, output_path: str) -> str:
        """保存配置文件。

        Parameters
        ----------
        config : dict
            配置字典。
        output_path : str
            输出目录路径。

        Returns
        -------
        str
            配置文件路径。
        """
        import json
        import os

        os.makedirs(output_path, exist_ok=True)
        config_path = os.path.join(output_path, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return config_path

    @staticmethod
    def _save_safetensors(
        state_dict: dict, output_path: str, filename: str
    ) -> str:
        """保存权重为 safetensors 格式。

        Parameters
        ----------
        state_dict : dict
            权重字典。
        output_path : str
            输出目录路径。
        filename : str
            文件名（不含路径）。

        Returns
        -------
        str
            权重文件路径。

        Raises
        ------
        ImportError
            safetensors 未安装。
        """
        import os

        os.makedirs(output_path, exist_ok=True)
        filepath = os.path.join(output_path, filename)

        try:
            from safetensors.torch import save_file  # type: ignore

            # 将 numpy array 转为 torch tensor
            # torch 是惰性导入
            import torch  # type: ignore

            torch_state = {}
            for k, v in state_dict.items():
                if not isinstance(v, torch.Tensor):
                    v = torch.as_tensor(v)
                torch_state[k] = v.contiguous()

            save_file(torch_state, filepath)
        except ImportError:
            raise ImportError(
                "safetensors is required to save weights in Mosaic standard format. "
                "Install via `pip install safetensors`."
            )

        return filepath
