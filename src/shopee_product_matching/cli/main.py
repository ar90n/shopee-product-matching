import os
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import git
import jupytext
import nbformat as nbf
import typer
import yaml
from dotenv import load_dotenv
from papermill.inspection import _infer_parameters as infer_parameters
from papermill.parameterize import parameterize_notebook, parameterize_path

from . import kaggle
from .build import build as kernel_build

load_dotenv(verbose=True)

app = typer.Typer()


def initialize_metadata(notebook: nbf.NotebookNode) -> None:
    notebook.metadata.papermill = {}
    for cell in notebook.cells:
        if not hasattr(cell.metadata, "tags"):
            cell.metadata.tags = []


def dump_params_as_yaml(params: List[Any]) -> str:
    param_dict = {p.name: eval(p.default) for p in params}
    return cast(str, yaml.dump(param_dict))


def param_to_str(params: Dict[str, Any]) -> str:
    ret = []
    for k, v in params.items():
        if isinstance(v, Iterable) and not isinstance(v, str):
            v = "[" + "_".join(str(p) for p in v) + "]"
        ret.append(f"{k}={v}")
    return "-".join(ret)


def split_key_value(key_value: str) -> Tuple[str, str]:
    [key, value] = key_value.split("=")[:2]
    return (key, value)


def split_key_value_with_eval(key_value: str) -> Tuple[str, Any]:
    [key, value] = split_key_value(key_value)
    try:
        eval_value = eval(value)
    except NameError:
        eval_value = value
    return (key, eval_value)


def get_param_names(code) -> List[str]:
    nb = jupytext.reads(code, fmt="py")
    initialize_metadata(nb)
    params = infer_parameters(nb)
    return [p.name for p in params]


def get_git_commit_hash() -> str:
    try:
        return git.repo.Repo(search_parent_directories=True).git.describe(
            always=True, dirty=True
        )
    except git.exc.InvalidGitRepositoryError:
        return ""


@app.command()
def build(
    input: Path,
    output: Optional[Path] = typer.Option(None),
    param_file: Optional[Path] = typer.Option(None),
    prologue: str = typer.Option("", envvar="SHOPEE_PRODUCT_MATCHING_PROLOGUE"),
    env: Optional[List[str]] = typer.Option(None),
    param: Optional[List[str]] = typer.Option(None),
    secret_key: Optional[List[str]] = typer.Option(None),
    use_internet: bool = True,
    strict: bool = False,
    memo: Optional[str] = None,
) -> None:
    git_commit_hash = get_git_commit_hash()
    if strict and "dirty" in git_commit_hash:
        raise EnvironmentError("git commit hash contains dirty")

    code = input.read_text()
    env_variables = {} if env is None else dict(split_key_value(e) for e in env)

    param_obj = (
        {} if param is None else dict(split_key_value_with_eval(e) for e in param)
    )
    if param_file is not None:
        param_obj = {**yaml.load(param_file.read_text()), **param_obj}

    if output is None:
        output = Path.cwd() / input.with_suffix(".ipynb").name
    if 0 < len(param_obj):
        param_info = param_to_str(param_obj)
        env_variables["NOTEBOOK_NAME"] = param_info
        output = output.with_name(f"{output.stem}-{param_info}{output.suffix}")

    secret_keys = list({*secret_key, "WANDB_API_KEY"})

    env_variables["EXPERIMENT_NAME"] = str(input.absolute().parent.stem)
    env_variables["GIT_COMMIT_HASH"] = git_commit_hash
    env_variables["PARAM_NAMES"] = ",".join(get_param_names(code))
    pkg_dataset = "shopeeproductmatchingrequirements"
    if memo is not None:
        env_variables["MEMO"] = memo

    nb = kernel_build(
        code,
        pkg_dataset=pkg_dataset,
        prologue=prologue,
        env_variables=env_variables,
        secret_keys=secret_keys,
        use_internet=use_internet,
    )
    initialize_metadata(nb)

    nb = parameterize_notebook(nb, param_obj)
    nb_str = jupytext.writes(nb, fmt="ipynb")
    output.write_text(nb_str)


@app.command()
def inspect(
    input: Path,
) -> None:
    code = input.read_text()
    nb = jupytext.reads(code, fmt="py")
    initialize_metadata(nb)

    params = infer_parameters(nb)
    print(dump_params_as_yaml(params))


@app.command()
def push(
    code_file: Path,
    slug: str = "shopee-product-matching",
    id_no: int = 16292473,
    title: str = "shopee product matching",
    is_public: bool = False,
    disable_gpu: bool = False,
    disable_internet: bool = False,
    dataset_source: Optional[List[str]] = typer.Option(None),
    kernel_source: Optional[List[str]] = typer.Option(None),
) -> None:
    dataset_sources = [] if dataset_source is None else list(dataset_source)
    kernel_sources = [] if kernel_source is None else list(kernel_source)
    competition_sources = ["shopee-product-matching"]
    dataset_sources.append("ar90ngas/shopeeproductmatchingrequirements")
    dataset_sources.append("ar90ngas/timm-pretrained-efficientnet")

    response = kaggle.push(
        id_no=id_no,
        slug=slug,
        title=title,
        body=code_file.read_text(),
        is_public=is_public,
        disable_gpu=disable_gpu,
        disable_internet=disable_internet,
        dataset_sources=dataset_sources,
        kernel_sources=kernel_sources,
        competition_sources=competition_sources,
    )
    kaggle.print_response(response)


@app.command()
def submit(
    ctx: typer.Context,
    input: Path,
    param_file: Optional[Path] = typer.Option(None),
    env: Optional[List[str]] = typer.Option(None),
    strict: bool = False,
) -> None:
    input = input.absolute()
    with TemporaryDirectory() as temp:
        output = Path(temp) / input.with_suffix(".ipynb").name
        ctx.invoke(
            build,
            input=input,
            output=output,
            param_file=param_file,
            prologue="",
            param={},
            secret_key=[],
            env=env,
            strict=strict,
            use_internet=False,
        )

        notebook_path = list(Path(temp).glob("*.ipynb"))[0]
        ctx.invoke(
            push,
            code_file=notebook_path,
            disable_internet=True,
            dataset_source=[],
            kernel_source=[],
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
