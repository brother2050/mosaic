# `Pipeline`

**模块**：`mosaic.core.pipeline`
**继承**：`Node`

## 描述

节点管道编排引擎。

继承 :class:`~mosaic.core.node.Node`，因此管道本身也是一种节点，可被
嵌套进更大的管道。

Parameters
----------
name:
    管道名称。
elements:
    有序的编排元素列表，每项为 ``Node``/``Pipeline``/``Branch``/``Merge``。
description:
    管道描述。

Examples
--------
基本用法::

    pipe = Pipeline("my-pipe", [
        TextGenerator(model="qwen2.5"),
        TextToImage(model="sdxl"),
        VideoEncoder(format="mp4"),
    ])
    result = pipe.run(input_data)

并行分支 + 合并::

    pipe = Pipeline("parallel", [
        ImageLoader(),
        Branch(
            bg=BackgroundRemover(),
            style=Stylizer(),
        ),
        Merge(),
    ])
    result = pipe.execute_result(input_data)
    # result.intermediate["bg"] → BackgroundRemover 输出
    # result.intermediate["style"] → Stylizer 输出

## 类属性

| 名称 | 值 |
|---|---|
| `name` | `'pipeline'` |
| `domain` | `'pipeline'` |
| `description` | `'A composable pipeline of nodes.'` |
| `version` | `'0.1.0'` |
| `input_types` | `[]` |
| `output_types` | `[]` |

## 方法

### `__call__(self, input_data: 'MosaicData') -> 'MosaicData'`

直接调用节点，等价于 :meth:`run`。

若节点尚未加载，会自动调用 :meth:`load`（惰性加载）。

### `__init__(self, name: 'str' = 'pipeline', elements: 'list[Any] | None' = None, description: 'str' = 'A composable pipeline of nodes.', **kwargs: 'Any') -> 'None'`

Initialize self.  See help(type(self)) for accurate signature.

### `accepts(self, data_type: 'str') -> 'bool'`

判断节点是否接受给定的数据类型标识。

### `add(self, element: 'Any') -> "'Pipeline'"`

追加一个编排元素，返回 ``self`` 以支持链式调用。

追加后已编译的 DAG 会失效，下次访问时重新编译。

### `describe(self) -> 'NodeSpec'`

返回管道的聚合规格说明。

### `dry_run(self) -> 'DryRunResult'`

干跑模式：只校验结构合法性与节点输入/输出类型匹配，不实际执行。

类型匹配规则：若前驱 ``output_types`` 与后继 ``input_types`` 无交集，
且二者均非空，则报告不匹配。空类型列表视为"接受任意类型"。

### `execute(self, input_data: 'MosaicData', *, config: 'RunConfig | None' = None, callbacks: 'list[EventHandler] | None' = None, context: 'Context | None' = None, fail_fast: 'bool' = True, max_workers: 'int' = 4) -> 'MosaicData'`

执行管道，返回最终输出 :class:`MosaicData`。

向后兼容的执行入口。如需获取完整运行信息（中间产物、错误列表、
耗时统计），请使用 :meth:`execute_result`。

Parameters
----------
input_data:
    管道输入数据。
config:
    运行配置（设备、精度、批大小等）。``None`` 使用默认配置。
callbacks:
    事件回调列表，每个回调会在节点开始/结束及管道级事件时被触发。
context:
    外部传入的运行上下文。``None`` 时创建新上下文。
fail_fast:
    某节点失败时是否立即抛出异常。``True``（默认）立即抛出；
    ``False`` 收集所有错误后返回（最终输出可能为 ``None``）。
max_workers:
    并行执行的最大线程数。Branch 的多条路径会并行执行。

Returns
-------
MosaicData
    管道最终输出。若存在多个终点，则返回以各终点标签为键、对应
    输出为值的 ``MosaicData``；单终点时直接返回该输出。
    ``fail_fast=False`` 且有节点失败时，最终输出可能为 ``None``
    （转为空 ``MosaicData``）。

### `execute_result(self, input_data: 'MosaicData', *, config: 'RunConfig | None' = None, callbacks: 'list[EventHandler] | None' = None, context: 'Context | None' = None, fail_fast: 'bool' = True, max_workers: 'int' = 4) -> 'PipelineResult'`

执行管道，返回完整的 :class:`PipelineResult`。

与 :meth:`execute` 相同的执行逻辑，但返回包含中间产物、错误列表、
各节点耗时的完整结果对象。

Parameters
----------
input_data:
    管道输入数据。
config:
    运行配置。``None`` 使用默认配置。
callbacks:
    事件回调列表。
context:
    外部传入的运行上下文。``None`` 时创建新上下文。
fail_fast:
    某节点失败时是否立即抛出异常。``True``（默认）立即抛出；
    ``False`` 收集所有错误，其他分支继续执行。
max_workers:
    并行执行的最大线程数。

Returns
-------
PipelineResult
    包含最终输出、中间产物、错误列表和耗时统计的完整结果。

### `get_intermediate(self, name: 'str') -> 'MosaicData'`

获取某节点的中间输出。

优先按节点 id 精确匹配；若未命中，则按节点 ``name`` 取首个匹配。

Raises
------
RuntimeError
    管道尚未运行。
KeyError
    找不到对应产物。

### `is_loaded(self) -> 'bool'`

检查模型是否已加载。

### `load(self) -> 'None'`

加载 DAG 中所有节点（含子管道）的模型。

### `produces(self) -> 'list[str]'`

返回节点输出的数据类型标识列表。

### `run(self, input_data: 'MosaicData', *, config: 'RunConfig | None' = None, callbacks: 'list[EventHandler] | None' = None, context: 'Context | None' = None) -> 'MosaicData'`

执行管道。

等价于 :meth:`execute`，便于以 ``Node`` 接口调用（嵌套场景）。

### `run_async(self, input_data: 'MosaicData', **kwargs: 'Any') -> "'AsyncTask'"`

异步执行管道，返回 :class:`~mosaic.core.task.AsyncTask`。

在新线程中调用 :meth:`execute_result`，不阻塞调用线程。
适用于视频生成等长时间运行的任务。

Parameters
----------
input_data:
    管道输入数据。
**kwargs:
    透传给 :meth:`execute_result` 的额外参数
    （如 ``config``、``fail_fast``、``max_workers``）。

Returns
-------
AsyncTask
    异步任务实例，可用于查询状态、等待结果、注册回调或取消。

Examples
--------
>>> task = pipe.run_async(input_data)
>>> task.status      # "pending" / "running" / "completed" / "failed"
>>> task.progress    # 0.0 - 1.0
>>> result = task.wait(timeout=300)

使用回调：
>>> task = pipe.run_async(input_data)
>>> task.on_complete(lambda r: print(f"Done: {r}"))
>>> task.on_error(lambda e: print(f"Error: {e}"))

### `unload(self) -> 'None'`

卸载 DAG 中所有节点的模型。

### `validate(self) -> 'None'`

校验 DAG 结构合法性。

Raises
------
PipelineError
    存在环、不可达节点或死端节点时抛出。
