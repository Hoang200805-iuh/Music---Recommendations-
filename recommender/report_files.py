"""Ghi báo cáo an toàn khi CSV đang được mở bằng Excel trên Windows."""
import os
from pathlib import Path
import tempfile
import time


def new_report_temp(target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    return Path(name)


def publish_report(staged, target):
    staged, target = Path(staged), Path(target)
    alternatives = [target, target.with_name(f"{target.stem}.latest{target.suffix}"),
                    target.with_name(f"{target.stem}.{time.time_ns()}{target.suffix}")]
    for candidate in alternatives:
        try:
            staged.replace(candidate)
        except PermissionError:
            continue
        if candidate != target:
            print(f"[report] Không ghi đè được CSV đang bận: {target}. Đã lưu: {candidate}", flush=True)
        return candidate
    raise PermissionError(f"Không ghi được báo cáo trong {target.parent}; dữ liệu tạm còn ở {staged}")


def write_report(target, writer):
    """Ghi hoàn chỉnh rồi mới thay file đích; không xóa file đang được người dùng mở."""
    staged = new_report_temp(target)
    try:
        writer(staged)
    except Exception:
        staged.unlink(missing_ok=True)
        raise
    return publish_report(staged, target)
