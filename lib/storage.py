"""集成层的基础设施：路径权威、原子 JSON 读写、哈希/摘要与输入快照工具。

只消费 paths 提供的路径权威；只维护集成诊断工件，原图与原 B 工件始终只读。
写出采用同目录临时文件后原子替换，避免读到半个 JSON，也不重发任何 API 请求。

主要定义：
- IntegrationError: 带机器可读 code 的集成层异常类型。
- project_path: 把相对路径按项目根解析为绝对路径。
- output_root: 返回集成输出的根目录。
- read_json: 以 utf-8-sig 读取 JSON 文件。
- write_json: 临时写后原子替换，含 Windows 占用重试。
- file_hash: 计算文件的 SHA256。
- digest: 对对象做稳定排序序列化后取 SHA256 摘要。
- fingerprint: 生成文件或目录内容快照摘要，不把目录名当样品真值。
- finite_number: 判断是否为范围内有限数值（排除布尔与 NaN/Inf）。

关联：
- paths: 提供 PROJECT_ROOT 与 UI_OUT_ROOT 路径权威。
- 本包其余模块（adapters、evidence_store、confidence_gate、reflection 等）统一复用上述工具。
"""
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
    # Windows原子替换瞬间可能拒绝并行读者；仅重试本地读，不重发任何模型请求。
    for attempt in range(4):
        try:
            with Path(path).open(encoding="utf-8-sig") as stream:
                return json.load(stream)
        except PermissionError as exc:
            if os.name != 'nt' or (getattr(exc,'winerror',None) not in {5,32,33}
                                  and exc.errno != 13) or attempt == 3:
                raise
            time.sleep(.01 * (2**attempt))


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
