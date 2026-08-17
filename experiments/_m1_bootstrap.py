"""experiments/_m1_bootstrap.py — E1-E3 的 M1 依赖解析（公开镜像，可复现）。

C1 的实验脚本运行在 M1 的液态模型（``mt_lnn`` 包）上。解析顺序：
  1. 兄弟目录 ``../M1``（本地开发布局，团队内使用）
  2. 缓存克隆（Windows: ``%LOCALAPPDATA%\\awareliquid\\M1``；
     POSIX: ``~/.cache/awareliquid/M1``）
  3. 自动浅克隆**公开镜像** https://github.com/AwareLiquid/M1.git
     （与 C1 同许可同公开策略，无需访问任何私有仓库）

用法（脚本内两行）：
    from _m1_bootstrap import bootstrap_m1
    bootstrap_m1()
"""

import os
import subprocess
import sys

_PUBLIC_M1_URL = "https://github.com/AwareLiquid/M1.git"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cache_dir() -> str:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
        return os.path.join(base, "awareliquid", "M1")
    return os.path.join(os.path.expanduser("~"), ".cache", "awareliquid", "M1")


def _prepend(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


def _is_m1_root(path: str) -> bool:
    return os.path.isdir(os.path.join(path, "mt_lnn"))


def bootstrap_m1() -> str:
    """确保 ``mt_lnn`` 可导入；返回 M1 仓库根目录（已加入 sys.path）。"""
    sibling = os.path.normpath(os.path.join(_REPO_ROOT, "..", "M1"))
    if _is_m1_root(sibling):
        _prepend(sibling)
        return sibling

    cached = _cache_dir()
    if _is_m1_root(cached):
        _prepend(cached)
        return cached

    os.makedirs(os.path.dirname(cached), exist_ok=True)
    print(f"[m1-bootstrap] cloning public mirror {_PUBLIC_M1_URL} -> {cached}")
    subprocess.run(["git", "clone", "--depth", "1", _PUBLIC_M1_URL, cached],
                   check=True)
    if not _is_m1_root(cached):
        raise RuntimeError(f"cloned {cached} but mt_lnn/ not found — "
                           "unexpected mirror layout")
    _prepend(cached)
    return cached


if __name__ == "__main__":
    root = bootstrap_m1()
    print(f"m1 root: {root}")
