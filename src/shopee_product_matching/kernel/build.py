from typing import Iterable, Dict
from .notebook import create_kernel, create_encoded_archive
from .setuptools import build_packages, get_dependencies


def build(
    code: str,
    pkg_dataset: str,
    prologue: str,
    env_variables: Dict[str, str],
    secret_keys: Iterable[str],
    enable_internet: bool,
) -> str:
    encoded_archive = create_encoded_archive(build_packages())
    dependencies = get_dependencies()
    return create_kernel(
        code,
        pkg_encoded=encoded_archive,
        pkg_dataset=pkg_dataset,
        env_variables=env_variables,
        dependencies=dependencies,
        secret_keys=secret_keys,
        prologue=prologue,
        enable_internet=enable_internet,
    )