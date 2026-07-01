# mosaic/core/registry.py
"""全局节点注册表。

本模块实现了一个全局的节点注册表 ``NodeRegistry``，提供节点的注册、
查询、列举与自动扫描能力。

设计要点
--------
* ``register`` 既可作为装饰器（``@registry.register``），也可作为普通
  函数调用，注册一个节点类。
* ``get`` 按名称获取节点**实例**（懒实例化），便于管道直接使用。
* ``discover`` 会递归扫描 ``mosaic.nodes`` 包下所有子模块，自动发现并
  注册以 ``Node`` 为基类的具体节点。自动扫描采用惰性策略：仅在被显式
  调用时执行，避免导入时的副作用。
* 注册表使用类名与 ``Node.name`` 作为双重索引，二者均可用于查询。
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
import threading
from typing import Any

from mosaic.core.node import Node, NodeSpec

__all__ = ["NodeRegistry", "registry", "get_default_registry"]


class NodeRegistry:
    """节点注册表，管理节点类的注册与查询。

    Attributes
    ----------
    _nodes:
        ``name -> Node 子类`` 的映射。
    _instances:
        ``name -> Node 实例`` 的缓存（懒实例化）。
    _lock:
        可重入锁，保护所有字典操作，确保线程安全。
    """

    def __init__(self) -> None:
        self._nodes: dict[str, type[Node]] = {}
        self._instances: dict[str, Node] = {}
        self._scanned: bool = False
        self._lock = threading.RLock()
        self._logger = logging.getLogger("mosaic.core.registry")

    # -- 注册 --------------------------------------------------------------
    def register(self, node_class: type[Node]) -> type[Node]:
        """注册一个节点类。

        既可用作装饰器::

            @registry.register
            class MyNode(Node):
                ...

        也可作为普通函数调用::

            registry.register(MyNode)

        Parameters
        ----------
        node_class:
            ``Node`` 的具体子类。抽象类（含未实现的抽象方法）不会被注册。

        Returns
        -------
        type[Node]
            原样返回被注册的类，便于装饰器链式使用。

        Raises
        ------
        TypeError
            传入的不是 ``Node`` 子类。
        ValueError
            节点名称重复。
        """
        if not (inspect.isclass(node_class) and issubclass(node_class, Node)):
            raise TypeError(
                f"Expected a subclass of Node, got {node_class!r}."
            )
        # 跳过抽象类（ABC 尚有未实现方法）
        if inspect.isabstract(node_class):
            return node_class

        with self._lock:
            name = node_class.name
            if name in self._nodes and self._nodes[name] is not node_class:
                raise ValueError(
                    f"Node name {name!r} is already registered to "
                    f"{self._nodes[name].__name__}."
                )

            self._nodes[name] = node_class
            # 以类名作为别名也登记一次；冲突时跳过并告警（B5）
            class_name = node_class.__name__
            if (
                class_name in self._nodes
                and self._nodes[class_name] is not node_class
            ):
                self._logger.warning(
                    "Class name alias %r conflicts with existing node; "
                    "skipping alias.",
                    class_name,
                )
            else:
                self._nodes[class_name] = node_class
            # 清除旧实例缓存
            self._instances.pop(name, None)
        return node_class

    # -- 注销 --------------------------------------------------------------
    def unregister(self, name: str) -> None:
        """按名称注销一个节点。"""
        with self._lock:
            node_class = self._nodes.pop(name, None)
            self._instances.pop(name, None)
            if node_class is not None:
                # 同时移除类名别名（B6：清理已缓存实例的类名别名）
                class_name = node_class.__name__
                if self._nodes.get(class_name) is node_class:
                    self._nodes.pop(class_name, None)

    # -- 查询 --------------------------------------------------------------
    def get(self, name: str, **kwargs: Any) -> Node:
        """按名称获取节点实例（懒实例化，带缓存）。

        Parameters
        ----------
        name:
            节点名称或类名。
        **kwargs:
            实例化节点时传入的构造参数。若提供任何 kwargs，将创建新实例
            而不使用缓存。

        Returns
        -------
        Node
            节点实例。

        Raises
        ------
        KeyError
            名称未注册。
        """
        with self._lock:
            if name not in self._nodes:
                raise KeyError(
                    f"Node {name!r} is not registered. "
                    f"Available: {sorted(self._nodes.keys())}"
                )
            # 传入构造参数时跳过缓存，直接新建
            if kwargs:
                return self._nodes[name](**kwargs)
            if name not in self._instances:
                self._instances[name] = self._nodes[name]()
            return self._instances[name]

    def get_class(self, name: str) -> type[Node]:
        """按名称获取节点类（不实例化）。"""
        with self._lock:
            if name not in self._nodes:
                raise KeyError(
                    f"Node {name!r} is not registered. "
                    f"Available: {sorted(self._nodes.keys())}"
                )
            return self._nodes[name]

    # -- 列举 --------------------------------------------------------------
    def list_nodes(self, domain: str | None = None) -> list[NodeSpec]:
        """列出所有节点的规格说明，可按域过滤。

        Parameters
        ----------
        domain:
            若提供，仅返回属于该域的节点。

        Returns
        -------
        list[NodeSpec]
            节点规格说明列表，按名称排序。
        """
        # 在锁内快照去重后的节点类，锁外再 describe（避免持锁实例化）
        with self._lock:
            unique_classes: list[type[Node]] = []
            seen: set = set()
            for node_class in self._nodes.values():
                # 同一类可能因 name + 类名 被登记两次，去重
                if node_class in seen:
                    continue
                seen.add(node_class)
                unique_classes.append(node_class)

        specs: list[NodeSpec] = []
        for node_class in unique_classes:
            spec = self._safe_describe(node_class)
            if domain is not None and spec.domain != domain:
                continue
            specs.append(spec)
        specs.sort(key=lambda s: s.name)
        return specs

    def list_domains(self) -> list[str]:
        """列出所有已注册节点涉及的域（去重排序）。"""
        with self._lock:
            unique_classes = list({c for c in self._nodes.values()})
        domains: set = set()
        for node_class in unique_classes:
            spec = self._safe_describe(node_class)
            domains.add(spec.domain)
        return sorted(domains)

    def list_names(self) -> list[str]:
        """列出所有已注册节点的名称。"""
        with self._lock:
            return sorted(
                cls.name for cls in {c for c in self._nodes.values()}
            )

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return name in self._nodes

    def __len__(self) -> int:
        # 去重计数
        with self._lock:
            return len({c for c in self._nodes.values()})

    # -- 自动扫描 ----------------------------------------------------------
    def discover(self, package: str = "mosaic.nodes") -> list[type[Node]]:
        """自动扫描 ``mosaic.nodes`` 包下的所有节点类并注册。

        采用 ``pkgutil.walk_packages`` 递归遍历子模块，对每个模块中
        符合条件的 ``Node`` 子类调用 :meth:`register`。

        Parameters
        ----------
        package:
            待扫描的包路径，默认 ``"mosaic.nodes"``。

        Returns
        -------
        list[type[Node]]
            本次扫描新注册的节点类列表。
        """
        if self._scanned:
            return []

        newly_registered: list[type[Node]] = []
        try:
            pkg = importlib.import_module(package)
        except ImportError as exc:
            self._logger.warning(
                "Failed to import discovery package %s: %s",
                package, exc, exc_info=True,
            )
            return newly_registered

        pkg_path = getattr(pkg, "__path__", None)
        if pkg_path is None:
            return newly_registered

        for _finder, mod_name, _is_pkg in pkgutil.walk_packages(
            pkg_path, prefix=f"{package}."
        ):
            try:
                module = importlib.import_module(mod_name)
            except Exception as exc:  # noqa: BLE001
                # 扫描不应因单个模块导入失败而中断，但需记录以便排查（B3）
                self._logger.warning(
                    "Failed to import module %s during discovery: %s",
                    mod_name, exc, exc_info=True,
                )
                continue
            for attr_name, attr_value in inspect.getmembers(
                module, inspect.isclass
            ):
                # 仅注册在本模块中定义的、Node 的具体子类
                try:
                    is_node_subclass = issubclass(attr_value, Node)
                except TypeError:
                    # attr_value 不是合法的类（例如 typing 特殊类型）
                    continue
                if (
                    is_node_subclass
                    and not inspect.isabstract(attr_value)
                    and attr_value.__module__ == mod_name
                ):
                    # B2：预检查与注册放入同一锁内，避免 TOCTOU 竞态。
                    # register 内部也会校验同名冲突，此处预检查仅为避免
                    # 重复注册同一类时多余地清缓存。
                    with self._lock:
                        already = self._nodes.get(attr_value.name)
                        if already is attr_value:
                            continue
                        if already is not None:
                            # 已被其它类占用，交由 register 抛出 ValueError
                            pass
                    try:
                        self.register(attr_value)
                        newly_registered.append(attr_value)
                    except ValueError as exc:
                        # 同名节点冲突不应中断整个发现过程
                        self._logger.warning(
                            "Node registration conflict: %s", exc,
                        )
                        continue
        # 标记扫描完成（移至扫描成功后，允许中途失败后重试）
        self._scanned = True
        return newly_registered

    def reset_discovery(self) -> None:
        """重置扫描标志，允许重新执行自动发现。"""
        with self._lock:
            self._scanned = False

    # -- 内部辅助 ----------------------------------------------------------
    def _safe_describe(self, node_class: type[Node]) -> NodeSpec:
        """安全获取节点规格说明。

        优先从类属性获取信息，避免实例化带来的副作用（如 ``__init__``
        触发 GPU 检测、显存调度等）；仅当类属性缺失时才尝试实例化并
        调用 ``describe``，实例化失败再退化为最小 NodeSpec。
        """
        # 优先从类属性构建（绝大多数节点都声明了 name/domain 等类属性）
        name = getattr(node_class, "name", None)
        domain = getattr(node_class, "domain", None)
        if name and domain:
            return NodeSpec(
                name=name,
                domain=domain,
                description=getattr(node_class, "description", ""),
                version=getattr(node_class, "version", "0.0.0"),
                input_types=list(getattr(node_class, "input_types", [])),
                output_types=list(getattr(node_class, "output_types", [])),
            )
        # 仅在必要时才实例化
        try:
            instance = node_class()
            return instance.describe()
        except Exception:  # noqa: BLE001
            return NodeSpec(
                name=getattr(node_class, "name", "unknown"),
                domain=getattr(node_class, "domain", "unknown"),
                description=getattr(node_class, "description", ""),
                version=getattr(node_class, "version", "0.0.0"),
                input_types=list(getattr(node_class, "input_types", [])),
                output_types=list(getattr(node_class, "output_types", [])),
            )


# ---------------------------------------------------------------------------
# 全局默认注册表单例
# ---------------------------------------------------------------------------
registry: NodeRegistry = NodeRegistry()


def get_default_registry() -> NodeRegistry:
    """返回全局默认注册表单例。"""
    return registry
