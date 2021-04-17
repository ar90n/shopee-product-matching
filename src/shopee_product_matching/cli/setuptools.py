import os
import types
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

from .package import Package


def get_dependencies() -> List[str]:
    requirements_txt_path = find_in_ancestors("requirements.txt")
    return [
        s
        for s in requirements_txt_path.read_text().split("\n")
        if not s.startswith("-e")
    ]


def find_in_ancestors(filename: str, cur_path: Path = Path.cwd()) -> Path:
    while True:
        candidate = cur_path / filename
        if candidate.exists():
            return candidate

        if cur_path == cur_path.parent:
            break
        cur_path = cur_path.parent

    raise EnvironmentError(f"{filename} is not found in ancestors.")


def build_wheel(setup_py: Path, dist_dir: Path, bdist_dir: Path) -> Path:
    code = setup_py.read_text()
    module = types.ModuleType("main")
    exec(code, module.__dict__)
    module.main(
        script_args=["bdist_wheel"],
        options={
            "bdist_wheel": {"dist_dir": str(dist_dir), "bdist_dir": str(bdist_dir)}
        },
    )
    try:
        it = dist_dir.glob("*.whl")
        return next(it)
    except IndexError:
        raise RuntimeError("failed to build wheel package.")


@contextmanager
def working_directory(path: Path):
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def build_packages() -> List[Package]:
    setup_py_path = find_in_ancestors("setup.py")
    with working_directory(setup_py_path.parent), TemporaryDirectory() as temp_dir_str:
        dist_dir = Path(temp_dir_str)
        bdist_dir = dist_dir / "build"
        pkg_path = build_wheel(setup_py_path, dist_dir, bdist_dir)
        pkg_bytes = pkg_path.read_bytes()
        pkgs = [Package(pkg_path.name, pkg_bytes)]

    return pkgs
