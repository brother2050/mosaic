# `_ConditionalNode`

**模块**：`mosaic.core.pipeline`
**继承**：`Node`

## 描述

条件分支的内部路由节点。

持有各路径对应的子 :class:`Pipeline`，运行时按 ``condition`` 选择
一条路径执行并返回其输出。在父 DAG 中表现为单个节点（单前驱、单后继）。

## 类属性

| 名称 | 值 |
|---|---|
| `name` | `'conditional'` |
| `domain` | `'core'` |
| `description` | `'Conditional branch router: select one path at runtime.'` |
| `version` | `'0.1.0'` |
| `input_types` | `['mosaic']` |
| `output_types` | `['mosaic']` |

## 方法

### `__call__(self, input_data: 'MosaicData') -> 'MosaicData'`

直接调用节点，等价于 :meth:`run`。

若节点尚未加载，会自动调用 :meth:`load`（惰性加载）。

### `__init__(self, paths: "dict[str, 'Pipeline']", condition: 'Callable[[MosaicData], str]') -> 'None'`

Initialize self.  See help(type(self)) for accurate signature.

### `accepts(self, data_type: 'str') -> 'bool'`

判断节点是否接受给定的数据类型标识。

### `describe(self) -> 'NodeSpec'`

返回节点规格说明。

### `is_loaded(self) -> 'bool'`

检查模型是否已加载。

### `load(self) -> 'None'`

加载所有候选路径的子管道。

### `produces(self) -> 'list[str]'`

返回节点输出的数据类型标识列表。

### `run(self, input_data: 'MosaicData') -> 'MosaicData'`

按条件选择路径并执行。

### `unload(self) -> 'None'`

卸载所有候选路径的子管道。
