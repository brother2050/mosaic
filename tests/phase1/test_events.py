# tests/phase1/test_events.py
"""Phase 1 事件总线测试。

覆盖 EventBus 的 on/off/emit、回调异常隔离、多个回调顺序与事件元数据。
"""

from __future__ import annotations

import pytest

from mosaic.core.events import EventBus, EventType, MosaicEvent, LoggingListener, get_event_bus


# ===========================================================================
# T_EVT_01: on 注册监听，事件触发时 callback 被调用
# ===========================================================================
class TestEventBusOnAndEmit:
    """事件总线 on/emit 基本测试。"""

    def test_on_registers_and_emit_calls_callback(self, fresh_bus):
        """T_EVT_01: on 注册监听，emit 触发回调。"""
        bus = fresh_bus
        received = []

        def handler(event):
            received.append(event)

        bus.on(EventType.NODE_START, handler)
        bus.emit(EventType.NODE_START, node_name="test-node")
        assert len(received) == 1
        assert received[0].event_type == EventType.NODE_START
        assert received[0].payload["node_name"] == "test-node"

    def test_on_returns_callback(self, fresh_bus):
        """T_EVT_01: on 返回回调函数。"""
        def handler(event):
            pass

        result = fresh_bus.on(EventType.NODE_START, handler)
        assert result is handler

    def test_emit_returns_event(self, fresh_bus):
        """T_EVT_01: emit 返回 MosaicEvent 对象。"""
        event = fresh_bus.emit(EventType.PIPELINE_START, pipeline_name="test")
        assert isinstance(event, MosaicEvent)
        assert event.event_type == EventType.PIPELINE_START

    def test_wildcard_subscription(self, fresh_bus):
        """T_EVT_01: 通配符订阅接收所有事件。"""
        received = []

        def handler(event):
            received.append(event)

        fresh_bus.on(EventType.ALL, handler)
        fresh_bus.emit(EventType.NODE_START, node_name="n1")
        fresh_bus.emit(EventType.NODE_COMPLETE, node_name="n1")
        assert len(received) == 2

    def test_subscriber_count(self, fresh_bus):
        """T_EVT_01: subscriber_count 返回正确数量。"""
        assert fresh_bus.subscriber_count() == 0

        def h1(e):
            pass

        fresh_bus.on(EventType.NODE_START, h1)
        assert fresh_bus.subscriber_count() == 1
        assert fresh_bus.subscriber_count(EventType.NODE_START) == 1
        assert fresh_bus.subscriber_count(EventType.NODE_COMPLETE) == 0


# ===========================================================================
# T_EVT_02: off 取消监听后不再触发
# ===========================================================================
class TestEventBusOff:
    """事件总线 off 取消订阅测试。"""

    def test_off_removes_callback(self, fresh_bus):
        """T_EVT_02: off 取消监听后回调不再触发。"""
        received = []

        def handler(event):
            received.append(event)

        fresh_bus.on(EventType.NODE_START, handler)
        result = fresh_bus.off(EventType.NODE_START, handler)
        assert result is True

        fresh_bus.emit(EventType.NODE_START, node_name="x")
        assert len(received) == 0

    def test_off_unknown_callback_returns_false(self, fresh_bus):
        """T_EVT_02: off 未知回调返回 False。"""
        def handler(event):
            pass

        result = fresh_bus.off(EventType.NODE_START, handler)
        assert result is False

    def test_off_unknown_event_type_returns_false(self, fresh_bus):
        """T_EVT_02: off 未知事件类型返回 False。"""
        def handler(event):
            pass

        fresh_bus.on(EventType.NODE_START, handler)
        result = fresh_bus.off("unknown_type", handler)
        assert result is False

    def test_clear_removes_all(self, fresh_bus):
        """T_EVT_02: clear 清除所有订阅。"""
        def h1(e):
            pass

        def h2(e):
            pass

        fresh_bus.on(EventType.NODE_START, h1)
        fresh_bus.on(EventType.NODE_COMPLETE, h2)
        assert fresh_bus.subscriber_count() == 2
        count = fresh_bus.clear()
        assert count == 2
        assert fresh_bus.subscriber_count() == 0

    def test_clear_specific_type(self, fresh_bus):
        """T_EVT_02: clear 指定类型清除。"""
        def h1(e):
            pass

        def h2(e):
            pass

        fresh_bus.on(EventType.NODE_START, h1)
        fresh_bus.on(EventType.NODE_COMPLETE, h2)
        count = fresh_bus.clear(EventType.NODE_START)
        assert count == 1
        assert fresh_bus.subscriber_count(EventType.NODE_START) == 0
        assert fresh_bus.subscriber_count(EventType.NODE_COMPLETE) == 1


# ===========================================================================
# T_EVT_03: callback 异常不影响管道运行
# ===========================================================================
class TestEventBusExceptionIsolation:
    """事件总线回调异常隔离测试。"""

    def test_callback_exception_not_propagated(self, fresh_bus):
        """T_EVT_03: 回调异常不向上传播。"""
        def bad_handler(event):
            raise RuntimeError("callback failure")

        fresh_bus.on(EventType.NODE_START, bad_handler)
        # emit 不应抛出异常
        event = fresh_bus.emit(EventType.NODE_START, node_name="test")
        assert event is not None

    def test_other_callbacks_still_fire(self, fresh_bus):
        """T_EVT_03: 一个回调异常不影响其他回调。"""
        received = []

        def bad_handler(event):
            raise RuntimeError("callback failure")

        def good_handler(event):
            received.append(event.event_type)

        fresh_bus.on(EventType.NODE_START, bad_handler)
        fresh_bus.on(EventType.NODE_START, good_handler)
        fresh_bus.emit(EventType.NODE_START, node_name="test")
        assert len(received) == 1
        assert received[0] == EventType.NODE_START

    def test_non_callable_on_raises(self, fresh_bus):
        """T_EVT_03: 注册非 callable 回调抛出 TypeError。"""
        with pytest.raises(TypeError, match="must be callable"):
            fresh_bus.on(EventType.NODE_START, "not a function")  # type: ignore


# ===========================================================================
# T_EVT_04: 多个 callback 按注册顺序触发
# ===========================================================================
class TestEventBusCallbackOrder:
    """事件总线回调顺序测试。"""

    def test_callbacks_fire_in_registration_order(self, fresh_bus):
        """T_EVT_04: 多个回调按注册顺序触发。"""
        order = []

        def first(event):
            order.append("first")

        def second(event):
            order.append("second")

        def third(event):
            order.append("third")

        fresh_bus.on(EventType.NODE_START, first)
        fresh_bus.on(EventType.NODE_START, second)
        fresh_bus.on(EventType.NODE_START, third)
        fresh_bus.emit(EventType.NODE_START, node_name="test")
        assert order == ["first", "second", "third"]

    def test_mixed_wildcard_and_specific_order(self, fresh_bus):
        """T_EVT_04: 通配符和特定类型回调按注册顺序触发。"""
        order = []

        def specific(event):
            order.append("specific")

        def wildcard(event):
            order.append("wildcard")

        fresh_bus.on(EventType.NODE_START, specific)
        fresh_bus.on(EventType.ALL, wildcard)
        fresh_bus.emit(EventType.NODE_START, node_name="test")
        # 特定类型先注册，应在前
        assert order == ["specific", "wildcard"]


# ===========================================================================
# T_EVT_05: 事件对象包含正确的元数据
# ===========================================================================
class TestEventObjectMetadata:
    """事件对象元数据测试。"""

    def test_event_has_timestamp(self, fresh_bus):
        """T_EVT_05: 事件对象包含时间戳。"""
        event = fresh_bus.emit(EventType.NODE_START, node_name="test")
        assert event.timestamp > 0
        assert isinstance(event.timestamp, float)

    def test_event_payload_contains_passed_data(self, fresh_bus):
        """T_EVT_05: 事件 payload 包含传入的数据。"""
        event = fresh_bus.emit(
            EventType.NODE_COMPLETE,
            node_name="test-node",
            duration=1.5,
            output_summary={"tokens": 100},
        )
        assert event.payload["node_name"] == "test-node"
        assert event.payload["duration"] == 1.5
        assert event.payload["output_summary"] == {"tokens": 100}

    def test_event_repr(self, fresh_bus):
        """T_EVT_05: 事件 repr 包含关键信息。"""
        event = fresh_bus.emit(EventType.NODE_START, node_name="test")
        r = repr(event)
        assert "MosaicEvent" in r
        assert "node_start" in r

    def test_event_type_all_types(self):
        """T_EVT_05: EventType.all_types() 返回所有事件类型。"""
        types = EventType.all_types()
        assert EventType.PIPELINE_START in types
        assert EventType.NODE_START in types
        assert EventType.NODE_ERROR in types
        assert EventType.MODEL_LOAD in types
        assert len(types) == 9


# ===========================================================================
# 补充：LoggingListener 与单例
# ===========================================================================
class TestLoggingListener:
    """LoggingListener 测试。"""

    def test_attach_and_detach(self, fresh_bus):
        """LoggingListener 可挂载和卸载。"""
        listener = LoggingListener()
        listener.attach(fresh_bus)
        assert fresh_bus.subscriber_count() > 0
        listener.detach()
        assert fresh_bus.subscriber_count() == 0

    def test_get_event_bus_returns_singleton(self):
        """get_event_bus 返回全局单例。"""
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2

    def test_repr(self, fresh_bus):
        """EventBus repr 包含订阅者数量。"""
        def h(e):
            pass

        fresh_bus.on(EventType.NODE_START, h)
        assert "subscribers=1" in repr(fresh_bus)