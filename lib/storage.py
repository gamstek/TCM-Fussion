"""只消费paths路径权威；管理集成诊断工件，原图和原B工件只读。"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

import paths


class IntegrationError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def project_path(value):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else Path(paths.PROJECT_ROOT) / p).resolve()


def output_root():
    return Path(paths.UI_OUT_ROOT) / "fusion"


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path, obj):
    """同目录临时写后replace，避免读到半个JSON；不覆盖B工件。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".write-", suffix=".json", dir=p.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        # Windows读者/索引器可能短暂占用目标文件；只重试本地原子替换，不重发任何API。
        for attempt in range(8):
            try:
                os.replace(temporary, p)
                break
            except PermissionError as exc:
                if getattr(exc,'winerror',None) not in {5,32,33} or attempt==7:
                    raise
                time.sleep(.01*(2**attempt))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def fingerprint(path):
    """输入内容快照；不把文件夹名当作样品真值。"""
    p = Path(path)
    if not p.exists():
        raise IntegrationError("INPUT_MISSING", f"输入不存在: {p}")
    files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
    if not files:
        raise IntegrationError("INPUT_EMPTY", f"输入目录为空: {p}")
    return digest([(x.name if p.is_file() else x.relative_to(p).as_posix(),
                    file_hash(x)) for x in files])


def finite_number(value, minimum=0, maximum=None):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= minimum
            and (maximum is None or value <= maximum))
