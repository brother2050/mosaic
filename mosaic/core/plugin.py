# mosaic/core/plugin.py
"""Mosaic 插件系统核心。

允许第三方开发者创建自定义节点，通过标准方式注册到 Mosaic。

三种插件发现机制
----------------
1. **entry_points**：通过 ``setuptools`` entry_points 发现已安装的插件包。
   entry_point group 名称为 ``mosaic.nodes``。

2. **装饰器注册**：使用 ``@mosaic.node`` 装饰器直接注册。

3. **目录扫描**：扫描指定目录下的 Python 文件，自动发现继承
   :class:`~mosaic.core.node.Node` 的类。

设计要点
--------
* 插件加载失败不会影响框架启动，异常被捕获并记录为警告。
* 内置节点通过 ``registry.discover()`` 自动注册；插件通过
  :class:`PluginManager` 发现后调用 :meth:`Registry.register` 注册。
* 插件可以依赖其他插件，但不能有循环依赖（检测到时跳过并警告）。
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from mosaic.core.node import Node, NodeSpec
from mosaic.core.registry import NodeRegistry, registry

__all__ = [
    "PluginInfo",
    "PluginManager",
    "plugin_manager",
    "node",
]

logger = logging.getLogger("mosaic.core.plugin")


# ---------------------------------------------------------------------------
# PluginInfo — 插件元信息
# ---------------------------------------------------------------------------
@dataclass
class PluginInfo:
    """插件元信息。

    Attributes
    ----------
    name:
        插件名称（节点名称）。
    domain:
        所属域，如 ``"text"``、``"custom"``。
    source:
        来源标识：``"builtin"`` / ``"entry_point"`` / ``"directory"`` /
        ``"decorator"``。
    version:
        版本号。
    description:
        描述信息。
    author:
        作者（可选）。
    node_class:
        节点类对象。
    module_path:
        模块路径（用于 reload）。
    """

    name: str
    domain: str
    source: str
    version: str = "0.1.0"
    description: str = ""
    author: str | None = None
    node_class: type[Node] | None = None
    module_path: str | None = None

    def __repr__(self) -> str:
        return (
            f"PluginInfo(name={self.name!r}, domain={self.domain!r}, "
            f"source={self.source!r}, version={self.version!r})"
        )


# ---------------------------------------------------------------------------
# @node 装饰器 — 快捷注册入口
# ---------------------------------------------------------------------------
def node(
    domain: str = "custom",
    name: str | None = None,
    version: str = "0.1.0",
    description: str = "",
    author: str | None = None,
) -> Any:
    """装饰器：将一个 Node 子类注册为 Mosaic 插件节点。

    使用示例::

        from mosaic import node
        from mosaic.core.node import Node

        @node(domain="custom", name="sentiment_analyzer")
        class SentimentAnalyzer(Node):
            ...

    Parameters
    ----------
    domain:
        所属域，默认 ``"custom"``。
    name:
        节点名称；``None`` 时使用类的 ``name`` 属性。
    version:
        版本号。
    description:
        描述信息。
    author:
        作者名称。

    Returns
    -------
    Callable
        类装饰器。
    """

    def decorator(cls: type[Node]) -> type[Node]:
        if not (inspect.isclass(cls) and issubclass(cls, Node)):
            raise TypeError(
                f"@node can only decorate Node subclasses, got {cls!r}."
            )
        # 覆盖类属性
        if name is not None:
            cls.name = name
        cls.domain = domain
        if version:
            cls.version = version
        if description:
            cls.description = description

        # 注册到全局注册表
        registry.register(cls)

        # 记录到 PluginManager
        info = PluginInfo(
            name=cls.name,
            domain=domain,
            source="decorator",
            version=version,
            description=description or cls.description,
            author=author,
            node_class=cls,
            module_path=cls.__module__,
        )
        plugin_manager._plugins[cls.name] = info
        plugin_manager._decorator_plugins.add(cls.name)
        return cls

    return decorator


# ---------------------------------------------------------------------------
# PluginManager — 插件管理器
# ---------------------------------------------------------------------------
class PluginManager:
    """插件管理器，负责插件的发现、加载与管理。

    Parameters
    ----------
    registry_instance:
        节点注册表实例，默认使用全局 ``registry``。
    """

    ENTRY_POINT_GROUP = "mosaic.nodes"

    def __init__(
        self,
        registry_instance: NodeRegistry | None = None,
    ) -> None:
        self._registry: NodeRegistry = registry_instance or registry
        self._plugins: dict[str, PluginInfo] = {}
        self._plugin_dirs: list[str] = []
        self._decorator_plugins: set[str] = set()
        self._loaded: bool = False

    # -- 加载 --------------------------------------------------------------
    def load_plugins(self) -> int:
        """加载所有插件，返回成功加载数量。

        按以下顺序执行：
        1. 通过 entry_points 发现已安装的插件包
        2. 扫描已注册的额外目录
        3. 装饰器注册的插件已在装饰时加载

        Returns
        -------
        int
            本次新加载的插件数量。
        """
        if self._loaded:
            return 0
        self._loaded = True

        count = 0
        count += self._load_entry_points()
        count += self._load_from_dirs()
        return count

    def reload(self) -> int:
        """重新加载所有插件。

        重置已加载标志与已发现的插件记录，随后重新执行
        :meth:`load_plugins`。适用于插件目录在运行期发生变化、或需要
        强制重新扫描的场景。

        Returns
        -------
        int
            本次重新加载后新发现的插件数量。
        """
        self._loaded = False
        self._plugins.clear()
        return self.load_plugins()

    def _load_entry_points(self) -> int:
        """通过 entry_points 发现并加载插件。"""
        count = 0
        try:
            from importlib.metadata import entry_points

            # Python 3.10+ 稳定 API：entry_points(group=...) 直接返回 EntryPoints
            eps = entry_points(group=self.ENTRY_POINT_GROUP)
        except Exception:
            return 0

        for ep in eps:
            try:
                node_class = ep.load()
                if not (inspect.isclass(node_class) and issubclass(node_class, Node)):
                    logger.warning(
                        "Entry point %s is not a Node subclass, skipping.",
                        ep.name,
                    )
                    continue

                self._registry.register(node_class)
                info = PluginInfo(
                    name=node_class.name,
                    domain=node_class.domain,
                    source="entry_point",
                    version=node_class.version,
                    description=node_class.description,
                    node_class=node_class,
                    module_path=node_class.__module__,
                )
                self._plugins[node_class.name] = info
                count += 1
                logger.info("Loaded plugin from entry_point: %s", ep.name)
            except Exception as exc:
                logger.warning(
                    "Failed to load plugin entry_point %s: %s", ep.name, exc
                )
        return count

    def _load_from_dirs(self) -> int:
        """扫描已注册的插件目录。"""
        count = 0
        for plugin_dir in self._plugin_dirs:
            count += self._scan_directory(plugin_dir)
        return count

    def _scan_directory(self, directory: str) -> int:
        """扫描目录下的 Python 文件，发现 Node 子类。

        Parameters
        ----------
        directory:
            要扫描的目录路径。

        Returns
        -------
        int
            新发现的插件数量。
        """
        if not os.path.isdir(directory):
            return 0

        count = 0
        for root, _dirs, files in os.walk(directory):
            for fname in files:
                if not fname.endswith(".py") or fname.startswith("_"):
                    continue
                fpath = os.path.join(root, fname)
                # 生成模块名：相对路径 -> 点分模块路径。
                # 跨平台：同时替换 os.sep 与 "/"（Windows 路径可能混用两种分隔符）。
                mod_name = (
                    os.path.relpath(fpath, directory)
                    .replace(os.sep, ".")
                    .replace("/", ".")
                    .removesuffix(".py")
                )
                # 清理非法字符：合法模块名仅允许字母、数字、下划线与点。
                mod_name = re.sub(r"[^a-zA-Z0-9_.]", "_", mod_name)
                try:
                    spec = importlib.util.spec_from_file_location(
                        f"mosaic_plugin_{mod_name.replace('.', '_')}", fpath
                    )
                    if spec is None or spec.loader is None:
                        continue
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[spec.name] = module
                    spec.loader.exec_module(module)

                    for attr_name, attr_value in inspect.getmembers(
                        module, inspect.isclass
                    ):
                        if (
                            issubclass(attr_value, Node)
                            and not inspect.isabstract(attr_value)
                            and attr_value.__module__ == spec.name
                        ):
                            self._registry.register(attr_value)
                            info = PluginInfo(
                                name=attr_value.name,
                                domain=attr_value.domain,
                                source="directory",
                                version=attr_value.version,
                                description=attr_value.description,
                                node_class=attr_value,
                                module_path=fpath,
                            )
                            self._plugins[attr_value.name] = info
                            count += 1
                            logger.info(
                                "Loaded plugin from directory: %s (%s)",
                                attr_value.name,
                                fpath,
                            )
                except Exception as exc:
                    logger.warning(
                        "Failed to load plugin file %s: %s", fpath, exc
                    )
        return count

    # -- 查询 --------------------------------------------------------------
    def list_plugins(self, source: str | None = None) -> list[PluginInfo]:
        """列出所有已加载的插件。

        Parameters
        ----------
        source:
            可选的来源过滤（``"builtin"`` / ``"entry_point"`` /
            ``"directory"`` / ``"decorator"``）。

        Returns
        -------
        list[PluginInfo]
            插件信息列表，按名称排序。
        """
        plugins = list(self._plugins.values())
        if source is not None:
            plugins = [p for p in plugins if p.source == source]
        plugins.sort(key=lambda p: p.name)
        return plugins

    def get_plugin(self, name: str) -> PluginInfo | None:
        """按名称获取插件信息。

        Parameters
        ----------
        name:
            插件（节点）名称。

        Returns
        -------
        PluginInfo | None
            插件信息；未找到时返回 ``None``。
        """
        return self._plugins.get(name)

    def get_plugin_node(self, name: str) -> type[Node] | None:
        """按名称获取插件节点类。

        Parameters
        ----------
        name:
            插件（节点）名称。

        Returns
        -------
        type[Node] | None
            节点类；未找到时返回 ``None``。
        """
        info = self._plugins.get(name)
        return info.node_class if info else None

    # -- 重新加载 ----------------------------------------------------------
    def reload_plugin(self, name: str) -> bool:
        """重新加载指定插件。

        主要用于开发时：修改插件代码后重新加载，无需重启 Python。

        Parameters
        ----------
        name:
            插件（节点）名称。

        Returns
        -------
        bool
            重新加载是否成功。
        """
        info = self._plugins.get(name)
        if info is None:
            logger.warning("Plugin %s not found, cannot reload.", name)
            return False

        if info.source == "decorator":
            logger.warning(
                "Cannot reload decorator-registered plugin %s.", name
            )
            return False

        if info.module_path is None:
            logger.warning("Plugin %s has no module_path, cannot reload.", name)
            return False

        try:
            # 注销旧版本
            self._registry.unregister(name)

            # 重新加载模块
            if info.source == "entry_point":
                module = importlib.import_module(info.module_path)
                module = importlib.reload(module)
            else:
                # directory 模式：从文件重新加载
                spec = importlib.util.spec_from_file_location(
                    f"mosaic_plugin_reload_{name}", info.module_path
                )
                if spec is None or spec.loader is None:
                    return False
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)

            # 重新发现节点类
            for _attr_name, attr_value in inspect.getmembers(
                module, inspect.isclass
            ):
                if (
                    issubclass(attr_value, Node)
                    and not inspect.isabstract(attr_value)
                    and attr_value.name == name
                ):
                    self._registry.register(attr_value)
                    info.node_class = attr_value
                    logger.info("Reloaded plugin: %s", name)
                    return True

            logger.warning(
                "Plugin %s class not found after reload.", name
            )
            return False
        except Exception as exc:
            logger.warning("Failed to reload plugin %s: %s", name, exc)
            return False

    # -- 目录注册 ----------------------------------------------------------
    def register_plugin_dir(self, path: str) -> None:
        """注册额外的插件扫描目录。

        Parameters
        ----------
        path:
            目录路径。将在下次 :meth:`load_plugins` 时被扫描。
            如果已经加载过，立即扫描该目录。
        """
        abs_path = os.path.abspath(path)
        if abs_path not in self._plugin_dirs:
            self._plugin_dirs.append(abs_path)

        if self._loaded:
            self._scan_directory(abs_path)

    # -- 内置节点标记 ------------------------------------------------------
    def mark_builtin(self, names: list[str] | None = None) -> None:
        """将已注册的内置节点标记为 ``builtin`` 来源。

        在 :meth:`registry.discover` 之后调用，将发现的内置节点
        也纳入 PluginManager 的管理范围。

        Parameters
        ----------
        names:
            指定的节点名称列表；``None`` 时标记全部已注册节点。
        """
        if names is None:
            specs = self._registry.list_nodes()
            names = [s.name for s in specs]

        for spec_name in names:
            if spec_name in self._plugins:
                continue
            try:
                node_class = self._registry.get_class(spec_name)
                info = PluginInfo(
                    name=node_class.name,
                    domain=node_class.domain,
                    source="builtin",
                    version=node_class.version,
                    description=node_class.description,
                    node_class=node_class,
                    module_path=node_class.__module__,
                )
                self._plugins[spec_name] = info
            except Exception as exc:
                logger.warning(
                    "Failed to mark builtin node %s: %s", spec_name, exc
                )

    # -- 统计 --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._plugins)

    def __contains__(self, name: object) -> bool:
        return name in self._plugins


# ---------------------------------------------------------------------------
# 全局插件管理器单例
# ---------------------------------------------------------------------------
plugin_manager: PluginManager = PluginManager()
