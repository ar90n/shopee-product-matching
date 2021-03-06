import base64
import gzip
import io
import json
import tarfile
from typing import Dict, Iterable

from shopee_product_matching.cli.package import Package

BOOTSTRAP_TEMPLATE: str = """def __bootstrap__():
    import sys
    import base64
    import gzip
    import tarfile
    import os
    import io
    import subprocess
    from pathlib import Path
    from tempfile import TemporaryDirectory
    try:
        from kaggle_secrets import UserSecretsClient
        from kaggle_web_client import BackendError
        has_kaggle_packages = True
    except ImportError:
        has_kaggle_packages = False
    # install required packages
    pkg_path_list = []
    def _find_pkgs(dir):
        children = list(dir.glob("*"))
        if 0 < len([p.name for p in children if p.name in ["pyproject.toml", "setup.py"]]):
            pkg_path_list.append(str(dir))
            return
        for p in children:
            if p.is_dir():
                _find_pkgs(p)
            else:
                pkg_path_list.append(str(p))
    print("------- found packages ------")
    if "{pkg_dataset}" != "":
        pip_pkgs_path = Path.cwd().parent / "input" / "{pkg_dataset}" / "pip"
        _find_pkgs(pip_pkgs_path)
    if 0 < len(pkg_path_list):
        for pkg in pkg_path_list:
            args = ["pip", "install", "--no-deps", pkg]
            try:
                output = subprocess.run(args, capture_output=True, encoding="utf-8", check=True).stdout
                print(output)
            except:
                pass
    if {use_internet} and 0 < len({dependencies}):
        for pkg in {dependencies}:
            args = ["pip", "install", pkg]
            try:
                output = subprocess.run(args, capture_output=True, encoding="utf-8", check=True).stdout
                print(output)
            except:
                pass
    print("------- finish package install ------")
    # this is base64 encoded source code
    tar_io = io.BytesIO(gzip.decompress(base64.b64decode("{pkg_encoded}")))
    with TemporaryDirectory() as temp_dir:
        with tarfile.open(fileobj=tar_io) as tar:
            for member in tar.getmembers():
                pkg_path = Path(temp_dir) / f"{{member.name}}"
                content_bytes = tar.extractfile(member).read()
                pkg_path.write_bytes(content_bytes)
                output = subprocess.run(["pip", "install", "--no-deps", pkg_path], capture_output=True, encoding="utf-8", check=True).stdout
                print(output)
    print("------- finish shopee-product-matching install ------")
    sys.path.append("/kaggle/working")
    # Add secrets to environment variables
    if {use_internet} and has_kaggle_packages:
        user_secrets = UserSecretsClient()
        for k in {secret_keys}:
            try:
                os.environ[k] = user_secrets.get_secret(k)
            except BackendError:
                pass
    # Update environment variables
    os.environ.update({env_variables})
    os.environ.update({{"USE_INTERNET": str({use_internet})}})
__bootstrap__()"""


def create_bootstrap_code(
    pkg_encoded: str,
    pkg_dataset: str,
    env_variables: Dict[str, str],
    dependencies: Iterable[str],
    secret_keys: Iterable[str],
    use_internet: bool = False,
) -> str:
    return BOOTSTRAP_TEMPLATE.format(
        pkg_encoded=pkg_encoded,
        pkg_dataset=pkg_dataset,
        env_variables=json.dumps(env_variables),
        dependencies=json.dumps(dependencies),
        secret_keys=json.dumps(secret_keys),
        use_internet=use_internet,
        encoding="utf8",
    )


def create_encoded_archive(pkgs: Iterable[Package]) -> str:
    tar_output = io.BytesIO()
    with tarfile.TarFile(fileobj=tar_output, mode="w") as tar:
        for pkg in pkgs:
            info = tarfile.TarInfo(name=pkg.name)
            info.size = len(pkg.content)
            tar.addfile(info, io.BytesIO(pkg.content))

    compressed = gzip.compress(tar_output.getvalue(), compresslevel=9)
    return base64.b64encode(compressed).decode("utf-8")


SCRIPT_TEMPLATE: str = """{prologue}
{bootstrap_code}
{script_body}
"""


def create_kernel(
    script_body: str,
    pkg_encoded: str,
    pkg_dataset: str,
    env_variables: Dict[str, str],
    dependencies: Iterable[str],
    secret_keys: Iterable[str],
    prologue: str,
    use_internet: bool = False,
) -> str:
    bootstrap_code = create_bootstrap_code(
        pkg_encoded=pkg_encoded,
        pkg_dataset=pkg_dataset,
        env_variables=env_variables,
        dependencies=dependencies,
        secret_keys=secret_keys,
        use_internet=use_internet,
    )
    return SCRIPT_TEMPLATE.format(
        bootstrap_code=bootstrap_code,
        script_body=script_body,
        prologue=prologue,
        encoding="utf8",
    )
