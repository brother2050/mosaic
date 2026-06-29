# `Merge`

**模块**：`mosaic.core.branch`
**继承**：`Node`

## 描述

合并节点：将多条上游输出合并为单个 :class:`MosaicData`。

在 DAG 中拥有多个前驱。运行时引擎会把各前驱输出按其路径名/节点名
组装成一个 ``MosaicData``（键为标签，值为对应输出），再交给本节点。

Parameters
----------
strategy:
    合并策略：

    * ``"dict"``（默认）：原样返回组装好的 ``MosaicData``，
      下游可通过 ``result["路径名"]`` 访问各分支输出。
    * ``"flatten"``：将各分支 ``MosaicData`` 的键值平铺合并到一个
      ``MosaicData``（后出现的键覆盖先前的）。
merge_fn:
    自定义合并函数，签名 ``(MosaicData) -> MosaicData``。提供时优先于
    ``strategy`` 生效。
keep:
    选择性合并：只保留指定分支名的结果，忽略其他分支。
    例如 ``Merge(keep="path_a")`` 只返回 ``path_a`` 分支的输出。
    为 ``None`` 时合并所有分支。

Note
----
若 ``Merge`` 只有一个前驱（例如位于条件分支之后），其行为退化为
透传：``"dict"`` 原样返回，``"flatten"`` 平铺该单一输入的键。

## 类属性

| 名称 | 值 |
|---|---|
| `name` | `'merge'` |
| `domain` | `'core'` |
| `description` | `'Fan-in: merge multiple upstream outputs into one MosaicData.'` |
| `version` | `'0.1.0'` |
| `input_types` | `['mosaic']` |
| `output_types` | `['mosaic']` |

## 方法

### `__call__(self, input_data: 'MosaicData') -> 'MosaicData'`

直接调用节点，等价于 :meth:`run`。

若节点尚未加载，会自动调用 :meth:`load`（惰性加载）。

### `__init__(self, strategy: 'str' = 'dict', merge_fn: 'Callable[[MosaicData], MosaicData] | None' = None, keep: 'str | None' = None, **kwargs: 'Any') -> 'None'`

Initialize self.  See help(type(self)) for accurate signature.

### `accepts(self, data_type: 'str') -> 'bool'`

判断节点是否接受给定的数据类型标识。

### `describe(self) -> 'NodeSpec'`

返回节点规格说明。

### `is_loaded(self) -> 'bool'`

检查模型是否已加载。

### `load(self) -> 'None'`

合并节点无需加载模型。

### `produces(self) -> 'list[str]'`

返回节点输出的数据类型标识列表。

### `run(self, input_data: 'MosaicData') -> 'MosaicData'`

执行合并。

``input_data`` 已由引擎组装为 ``{标签: 分支输出}`` 形式。

### `unload(self) -> 'None'`

释放资源。
