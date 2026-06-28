# mosaic/core/node.py
"""节点抽象基类与节点规格说明。

本模块定义了 Mosaic 框架中所有 39 个节点必须继承的 ``Node`` 抽象基类，
以及描述节点元信息的 ``NodeSpec`` 数据结构。

设计要点
--------
* ``Node`` 采用模板方法风格：子类只需实现 ``load``/``unload``/``run``
  三个核心方法，其余能力（上下文管理、调用语法、加载状态）由基类提供。
* ``NodeSpec`` 是节点的"说明书"，向管道与注册表声明节点接受的输入
  类型、输出的类型以及模型信息。
* 节点通过类属性 ``input_types`` / ``output_types`` 声明数据契约，
  供管道在编排时做静态校验。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from mosaic.core.types import MosaicData

__all__ = ["NodeSpec", "Node"]


# ---------------------------------------------------------------------------
# NodeSpec — 节点规格说明
# ---------------------------------------------------------------------------
@dataclass
class NodeSpec:
    """节点的规格说明（元信息）。

    Attributes
    ----------
    name:
        节点唯一名称，如 ``"txt2img"``。
    domain:
        所属域，如 ``"image"``、``"text"``。
    description:
        节点功能描述。
    version:
        节点版本号。
    input_types:
        节点接受的数据类型列表（``MosaicData`` 子类的字符串标识）。
    output_types:
        节点输出的数据类型列表。
    model_info:
        模型相关信息字典，可包含：
        * ``name``：模型名称
        * ``source``：模型来源（如 HuggingFace repo id）
        * ``license``：模型许可证
        * ``vram_gb``：预估显存需求（GB）
    """

    name: str
    domain: str
    description: str = ""
    version: str = "0.1.0"
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    model_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """将规格说明转为纯字典。"""
        return {
            "name": self.name,
            "domain": self.domain,
            "description": self.description,
            "version": self.version,
            "input_types": list(self.input_types),
            "output_types": list(self.output_types),
            "model_info": dict(self.model_info),
        }

    def __repr__(self) -> str:
        return (
            f"NodeSpec(name={self.name!r}, domain={self.domain!r}, "
            f"version={self.version!r}, "
            f"input={self.input_types!r} -> output={self.output_types!r})"
        )


# ---------------------------------------------------------------------------
# Node — 节点抽象基类
# ---------------------------------------------------------------------------
class Node(abc.ABC):
    """所有 Mosaic 节点的抽象基类。

    子类必须实现以下抽象方法：
        * :meth:`load`    — 加载模型到 GPU/内存
        * :meth:`unload`  — 卸载模型，释放资源
        * :meth:`run`     — 执行节点逻辑
        * :meth:`describe` — 返回节点规格说明

    同时，子类应通过类属性声明数据契约：
        * ``name``        — 节点名称
        * ``domain``      — 所属域
        * ``description`` — 功能描述
        * ``version``     — 节点版本
        * ``input_types`` — 接受的数据类型标识列表
        * ``output_types``— 输出的数据类型标识列表

    示例
    -----
    >>> class MyNode(Node):
    ...     name = "my-node"
    ...     domain = "text"
    ...     description = "示例节点"
    ...     version = "0.1.0"
    ...     input_types = ["text"]
    ...     output_types = ["text"]
    ...
    ...     def load(self) -> None:
    ...         self._loaded = True
    ...
    ...     def unload(self) -> None:
    ...         self._loaded = False
    ...
    ...     def run(self, input_data: MosaicData) -> MosaicData:
    ...         return input_data
    ...
    ...     def describe(self) -> NodeSpec:
    ...         return NodeSpec(name=self.name, domain=self.domain,
    ...                         description=self.description, version=self.version,
    ...                         input_types=self.input_types,
    ...                         output_types=self.output_types)
    """

    # -- 类属性：数据契约（子类覆写）--------------------------------------
    name: str = "base-node"
    domain: str = "core"
    description: str = "Base node, override in subclass."
    version: str = "0.1.0"
    input_types: List[str] = []
    output_types: List[str] = []

    def __init__(self, **kwargs: Any) -> None:
        # 允许实例化时覆写类属性
        for key, value in kwargs.items():
            if hasattr(type(self), key):
                setattr(self, key, value)
        self._loaded: bool = False

    # -- 抽象方法（子类必须实现）------------------------------------------
    @abc.abstractmethod
    def load(self) -> None:
        """加载模型到 GPU/内存。

        实现应完成模型权重下载、设备迁移等初始化工作，并将内部
        ``_loaded`` 标志置为 ``True``。
        """

    @abc.abstractmethod
    def unload(self) -> None:
        """卸载模型，释放资源。

        实现应释放显存/内存占用，并将内部 ``_loaded`` 标志置为 ``False``。
        """

    @abc.abstractmethod
    def run(self, input_data: MosaicData) -> MosaicData:
        """执行节点逻辑。

        Parameters
        ----------
        input_data:
            输入数据容器，类型应符合 :attr:`input_types` 声明。

        Returns
        -------
        MosaicData
            输出数据容器，类型应符合 :attr:`output_types` 声明。
        """

    @abc.abstractmethod
    def describe(self) -> NodeSpec:
        """返回节点规格说明。"""

    # -- 便利方法 ----------------------------------------------------------
    def __call__(self, input_data: MosaicData) -> MosaicData:
        """直接调用节点，等价于 :meth:`run`。

        若节点尚未加载，会自动调用 :meth:`load`（惰性加载）。
        """
        if not self.is_loaded():
            self.load()
        return self.run(input_data)

    def __enter__(self) -> "Node":
        """进入上下文：自动加载模型。"""
        self.load()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """退出上下文：自动卸载模型。"""
        self.unload()
        # 不吞掉异常
        return None

    def is_loaded(self) -> bool:
        """检查模型是否已加载。"""
        return self._loaded

    # -- 输入类型校验（可选，供管道调用）----------------------------------
    def accepts(self, data_type: str) -> bool:
        """判断节点是否接受给定的数据类型标识。"""
        return data_type in self.input_types

    def produces(self) -> List[str]:
        """返回节点输出的数据类型标识列表。"""
        return list(self.output_types)

    # -- 管道运算符 --------------------------------------------------------
    def __or__(self, other: Any) -> Any:
        """支持 ``node_a | node_b`` 声明式管道语法。

        * ``Node | Node``     → 包含两个节点的匿名 :class:`Pipeline`
        * ``Node | Pipeline`` → 将本节点前置到目标管道首部（扁平展开）
        * ``Node | Branch``   → 包含本节点与分支的匿名 :class:`Pipeline`

        Parameters
        ----------
        other:
            右操作数，须为 ``Node``/``Pipeline``/``Branch``。

        Returns
        -------
        Pipeline
            匿名管道。若 ``other`` 类型不支持则返回 ``NotImplemented``。
        """
        # 延迟导入以避免与 pipeline 模块的循环依赖
        from mosaic.core.pipeline import Branch, Pipeline

        if isinstance(other, Pipeline):
            return Pipeline("anonymous", [self, *other.elements])
        if isinstance(other, Branch):
            return Pipeline("anonymous", [self, other])
        if isinstance(other, Node):
            return Pipeline("anonymous", [self, other])
        return NotImplemented

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"domain={self.domain!r} state={status}>"
        )
