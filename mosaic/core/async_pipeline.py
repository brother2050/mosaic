# mosaic/core/async_pipeline.py
"""异步管道执行桥接。

提供 :func:`create_async_task` 工厂函数，将 :class:`~mosaic.core.pipeline.Pipeline`
的同步执行封装为 :class:`~mosaic.core.task.AsyncTask`。

设计要点
--------
* :class:`Pipeline.run_async` 委托本模块创建并启动 ``AsyncTask``。
* ``AsyncTask`` 在独立线程中调用 ``pipeline.execute_result()``，
  同时通过 ``EventBus`` 订阅进度事件。
* 取消机制通过 ``context.shared["_cancel_event"]`` 传递
  ``threading.Event``，节点可协作式检查。
"""

from __future__ import annotations

from typing import Any

from mosaic.core.events import EventBus, get_event_bus
from mosaic.core.task import AsyncTask

__all__ = ["create_async_task"]


def create_async_task(
    pipeline: Any,
    input_data: Any,
    task_id: str | None = None,
    bus: EventBus | None = None,
    start: bool = True,
    **kwargs: Any,
) -> AsyncTask:
    """创建（并可选启动）异步管道执行任务。

    Parameters
    ----------
    pipeline:
        要执行的 :class:`Pipeline` 实例。
    input_data:
        管道输入数据。
    task_id:
        任务 ID，``None`` 自动生成。
    bus:
        事件总线实例，``None`` 使用全局单例。
    start:
        是否立即启动任务。``True``（默认）时创建独立裸线程启动，
        适用于 ``Pipeline.run_async()`` 等独立执行场景。
        ``False`` 时仅创建任务实例，由调用者（如
        :class:`~mosaic.core.task_manager.TaskManager`）通过
        ``task._start(executor=...)`` 注入共享线程池后再启动。
    **kwargs:
        透传给 ``pipeline.execute_result()`` 的额外参数
        （如 ``config``、``fail_fast``、``max_workers``）。

    Returns
    -------
    AsyncTask
        异步任务实例。``start=True`` 时已启动，``start=False`` 时尚未启动。

    Examples
    --------
    >>> task = create_async_task(pipeline, input_data)
    >>> task.on_complete(lambda r: print(f"Done: {r}"))
    >>> result = task.wait(timeout=300)
    """
    # 获取管道名称
    pipeline_name = getattr(pipeline, "name", "pipeline")

    # 创建任务
    task = AsyncTask(
        pipeline_name=pipeline_name,
        pipeline=pipeline,
        input_data=input_data,
        task_id=task_id,
        bus=bus or get_event_bus(),
        **kwargs,
    )

    # 启动任务（在新线程中执行）
    if start:
        task._start()
    return task
