# mosaic/core/task_manager.py
"""任务管理器。

管理所有 :class:`~mosaic.core.task.AsyncTask` 的生命周期，提供
提交、查询、取消、清理和统计功能。

设计要点
--------
* 使用 Python 标准库 ``threading`` 实现线程安全。
* 所有方法线程安全（内部 ``threading.Lock`` 保护）。
* ``submit()`` 通过共享 :class:`ThreadPoolExecutor` 提交任务，
  线程复用、并发受控（``max_workers`` 限制）、超限自动排队。
* ``cleanup()`` 清理已完成的过期任务，防止内存泄漏。
* ``status_summary()`` 返回各状态的任务数量统计。
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from mosaic.core.events import EventBus, get_event_bus
from mosaic.core.task import AsyncTask, TaskStatus

__all__ = ["TaskManager"]

# 默认最大并发任务数
_DEFAULT_MAX_WORKERS = 4


class TaskManager:
    """异步任务管理器。

    管理所有 ``AsyncTask`` 的生命周期，提供统一的提交、查询、
    取消和清理接口。内部使用共享线程池，线程复用且并发受控。

    Parameters
    ----------
    bus:
        事件总线实例，``None`` 使用全局单例。
    max_workers:
        线程池最大并发数，默认 4。超过此数的任务自动排队等待。
        设为 ``None`` 使用默认值。

    Examples
    --------
    >>> manager = TaskManager(max_workers=8)
    >>> task = manager.submit(pipeline, input_data)
    >>> task_id = task.task_id
    >>> # 查询状态
    >>> task = manager.get(task_id)
    >>> task.status
    'running'
    >>> # 列出所有运行中的任务
    >>> running = manager.list_tasks(status="running")
    >>> # 取消任务
    >>> manager.cancel(task_id)
    True
    >>> # 统计
    >>> manager.status_summary()
    {'pending': 0, 'running': 0, 'completed': 1, 'failed': 0, 'cancelled': 1, 'total': 2}
    """

    def __init__(
        self,
        bus: EventBus | None = None,
        max_workers: int | None = None,
    ) -> None:
        self._bus: EventBus = bus or get_event_bus()
        self._tasks: dict[str, AsyncTask] = {}
        self._lock: threading.Lock = threading.Lock()
        self._logger: logging.Logger = logging.getLogger("mosaic.task_manager")

        # 共享线程池：线程复用、并发受控、超限排队
        workers = max_workers or _DEFAULT_MAX_WORKERS
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="mosaic-task",
        )
        self._max_workers: int = workers
        self._logger.info(
            "TaskManager initialized with max_workers=%d.", workers,
        )

    # ------------------------------------------------------------------
    # 提交任务
    # ------------------------------------------------------------------
    def submit(
        self,
        pipeline: Any,
        input_data: Any,
        **kwargs: Any,
    ) -> AsyncTask:
        """提交管道异步执行任务。

        Parameters
        ----------
        pipeline:
            要执行的 :class:`Pipeline` 实例。
        input_data:
            管道输入数据。
        **kwargs:
            透传给 ``pipeline.execute_result()`` 的额外参数。

        Returns
        -------
        AsyncTask
            已启动的异步任务实例。
        """
        from mosaic.core.async_pipeline import create_async_task

        # start=False：不在 create_async_task 内部启动（裸线程），
        # 由下方 _start(executor=...) 注入共享线程池后统一启动。
        task = create_async_task(
            pipeline=pipeline,
            input_data=input_data,
            bus=self._bus,
            start=False,
            **kwargs,
        )

        with self._lock:
            self._tasks[task.task_id] = task

        # 通过共享线程池启动任务（复用线程、并发受控、超限排队）
        task._start(executor=self._executor)

        self._logger.info(
            "Submitted task %s (pipeline=%s).",
            task.task_id,
            task.pipeline_name,
        )
        return task

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get(self, task_id: str) -> AsyncTask | None:
        """获取指定任务。

        Parameters
        ----------
        task_id:
            任务 ID。

        Returns
        -------
        AsyncTask | None
            任务实例，不存在返回 ``None``。
        """
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(
        self,
        status: str | None = None,
    ) -> list[AsyncTask]:
        """列出所有任务，可按状态过滤。

        Parameters
        ----------
        status:
            状态过滤，``None`` 返回全部。可选值：
            ``pending`` / ``running`` / ``completed`` / ``failed`` / ``cancelled``。

        Returns
        -------
        list[AsyncTask]
            任务列表，按创建时间排序。
        """
        with self._lock:
            tasks = list(self._tasks.values())

        if status is not None:
            tasks = [t for t in tasks if t.status == status]

        # 按创建时间排序
        tasks.sort(key=lambda t: t.created_at)
        return tasks

    # ------------------------------------------------------------------
    # 取消
    # ------------------------------------------------------------------
    def cancel(self, task_id: str) -> bool:
        """取消指定任务。

        Parameters
        ----------
        task_id:
            任务 ID。

        Returns
        -------
        bool
            ``True`` 表示成功设置取消标志；
            ``False`` 表示任务不存在或已到达终态。
        """
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            return False
        return task.cancel()

    def cancel_all(self) -> int:
        """取消所有运行中的任务。

        Returns
        -------
        int
            成功取消的任务数量。
        """
        count = 0
        with self._lock:
            tasks = list(self._tasks.values())
        for task in tasks:
            if task.cancel():
                count += 1
        if count > 0:
            self._logger.info("Cancelled %d running task(s).", count)
        return count

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------
    def cleanup(self, max_age: float = 3600) -> int:
        """清理过期的已完成任务。

        删除已完成（completed/failed/cancelled）且创建时间超过
        ``max_age`` 秒的任务。

        Parameters
        ----------
        max_age:
            最大保留时间（秒），默认 3600 秒（1 小时）。

        Returns
        -------
        int
            清理的任务数量。
        """
        now = time.time()
        cutoff = now - max_age
        removed: list[str] = []

        with self._lock:
            for task_id, task in list(self._tasks.items()):
                # 只清理已完成的任务
                if task.status not in TaskStatus.terminal_statuses():
                    continue
                # 检查是否过期
                completed_at = task.completed_at or task.created_at
                if completed_at < cutoff:
                    removed.append(task_id)
                    del self._tasks[task_id]

        if removed:
            self._logger.info(
                "Cleaned up %d expired task(s).", len(removed)
            )
        return len(removed)

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    def status_summary(self) -> dict[str, int]:
        """返回任务统计信息。

        Returns
        -------
        dict[str, int]
            各状态的任务数量，包含 ``total`` 总数。
        """
        with self._lock:
            tasks = list(self._tasks.values())

        summary: dict[str, int] = {
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "total": len(tasks),
        }
        for task in tasks:
            s = task.status
            if s in summary:
                summary[s] += 1
        return summary

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def shutdown(self, wait: bool = True) -> None:
        """关闭线程池，释放资源。

        Parameters
        ----------
        wait:
            是否等待所有运行中的任务完成。
        """
        self._executor.shutdown(wait=wait)
        self._logger.info("TaskManager executor shutdown (wait=%s).", wait)

    def __enter__(self) -> "TaskManager":
        """上下文管理器入口。"""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """上下文管理器出口：自动关闭线程池。"""
        self.shutdown(wait=False)

    def __del__(self) -> None:
        """析构时安全关闭线程池（兜底，不保证调用时机）。"""
        try:
            executor = getattr(self, "_executor", None)
            if executor is not None:
                executor.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # 表示
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        with self._lock:
            n = len(self._tasks)
        return f"<TaskManager tasks={n} max_workers={self._max_workers}>"
