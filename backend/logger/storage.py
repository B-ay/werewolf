"""回放与竞技报告的原子 JSON 写入工具。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json_atomic(
    path: Path | str,
    payload: Any,
    *,
    temporary_path: Path | str | None = None,
) -> None:
    """先写同目录临时文件再替换成品，读者不会读到半截 JSON。

    调用方负责准备可序列化数据和授权写入路径。该函数不读取角色记忆或 API Key；
    回放由服务端提供裁判事件与评测，竞技报告由 TournamentRunner 提供。
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(temporary_path) if temporary_path is not None else destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
