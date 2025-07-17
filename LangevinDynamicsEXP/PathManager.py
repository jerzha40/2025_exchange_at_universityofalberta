"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
$祷言PathManager.py:
挖穿一切障碍
dig through every obstacles.
"""

"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
from pathlib import Path
import inspect
from typing import Union


class PathManager:
    _base_dir: Union[None, Path] = None
    _script_name: Union[None, str] = None

    def __init__(self) -> None:
        frame = inspect.stack()[1]
        caller_path = Path(frame.filename).resolve()
        PathManager._script_name = caller_path.stem

        if PathManager._base_dir is None:
            PathManager._base_dir = caller_path.parent / "simlog"
            PathManager._base_dir.mkdir(parents=True, exist_ok=True)

        self._dir: Path = PathManager._base_dir / PathManager._script_name
        self._dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def set_base_dir(cls, base_dir: Union[str, Path]) -> None:
        base = Path(base_dir)
        if not base.is_absolute():
            caller_path = Path(inspect.stack()[1].filename).resolve()
            base = (caller_path.parent / base).resolve()

        cls._base_dir = base
        cls._base_dir.mkdir(parents=True, exist_ok=True)

    def get_log_path(self) -> Path:
        assert PathManager._base_dir is not None, "Base directory is not initialized"
        assert PathManager._script_name is not None, "Script name is not set"
        return PathManager._base_dir / f"{PathManager._script_name}.log"

    def get_data_path(self, name: str, ext: str = ".json") -> Path:
        return self._dir / f"{name}{ext}"


if __name__ == "__main__":
    pm = PathManager()
    print("日志路径：", pm.get_log_path())
    print("保存账户数据：", pm.get_data_path("account"))
