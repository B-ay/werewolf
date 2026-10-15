"""只追加的裁判事件流；读取时按视角过滤并重新连续编号。"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ..game.models import GameEvent


class EventLogger:
    """每局独立的事件记录器，不依赖 Agent、模型、HTTP或评测器。

    事件保留真实裁判序号用于全知回放。普通玩家读取时只返回授权事件，并重编
    连续序号，避免从缺口统计其他角色私有行动。to_dict 深拷贝嵌套载荷，防止
    接收方篡改只追加的事实源。游戏状态及玩家是否合法仍由规则引擎检查。
    """

    def __init__(self) -> None:
        self.events: list[GameEvent] = []

    def append(
        self,
        *,
        round: int,
        phase: str,
        event_type: str,
        message: str,
        actor_id: Optional[str] = None,
        target_id: Optional[str] = None,
        data: Optional[Mapping[str, Any]] = None,
        visibility: Sequence[str] = ("public",),
    ) -> GameEvent:
        """按原顺序写入一个事件；完整事件仅保存在本局裁判日志中。"""

        event = GameEvent(
            seq=len(self.events) + 1,
            round=round,
            phase=phase,
            event_type=event_type,
            actor_id=actor_id,
            target_id=target_id,
            message=message,
            data=dict(data or {}),
            visibility=tuple(visibility),
        )
        self.events.append(event)
        return event

    def project(
        self, viewer_id: Optional[str] = None, *, omniscient: bool = False
    ) -> list[dict[str, Any]]:
        """生成全知、公共或指定玩家可见的独立序列化视图。"""

        if omniscient:
            return [event.to_dict(include_visibility=True) for event in self.events]
        selected = [event for event in self.events if event.visible_to(viewer_id)]
        result: list[dict[str, Any]] = []
        for visible_seq, event in enumerate(selected, start=1):
            item = event.to_dict(include_visibility=False)
            item["seq"] = visible_seq
            result.append(item)
        return result
