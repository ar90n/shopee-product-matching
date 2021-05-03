from pathlib import Path
from typing import Dict, List, Optional, cast

import git
import typer
from dotenv import load_dotenv
from kaggle.models.kaggle_models_extended import KernelPushResponse
from shopee_product_matching.cli import kaggle
from shopee_product_matching.cli.build import build as kernel_build

load_dotenv(verbose=True)

app = typer.Typer()


def _get_git_commit_hash() -> str:
    try:
        return cast(
            str,
            git.repo.Repo(search_parent_directories=True).git.describe(
                always=True, dirty=True
            ),
        )
    except git.exc.InvalidGitRepositoryError:
        return ""


def _parse_extra_args(args: List[str]) -> Dict[str, str]:
    envs = {}
    args.reverse()
    while 0 < len(args):
        k = args.pop()
        if not k.startswith("--"):
            continue
        k = k[2:]

        if not args[-1].startswith("--"):
            v = args.pop()
            envs[k] = v
    return envs


def _build(
    input: Path,
    prologue: str = "",
    envs: Optional[Dict[str, str]] = None,
    secret_keys: Optional[List[str]] = None,
    use_internet: bool = True,
    strict: bool = False,
    memo: Optional[str] = None,
) -> str:
    git_commit_hash = _get_git_commit_hash()
    if strict and "dirty" in git_commit_hash:
        raise EnvironmentError("git commit hash contains dirty")

    code = input.read_text()
    env_variables = {} if envs is None else envs
    env_variables["EXPERIMENT_NAME"] = str(input.absolute().parent.stem)
    env_variables["GIT_COMMIT_HASH"] = git_commit_hash
    secret_keys = [] if secret_keys is None else secret_keys
    secret_keys.append("WANDB_API_KEY")

    pkg_dataset = "shopeeproductmatchingrequirements"
    if memo is not None:
        env_variables["MEMO"] = memo

    return kernel_build(
        code,
        pkg_dataset=pkg_dataset,
        prologue=prologue,
        env_variables=env_variables,
        secret_keys=secret_keys,
        use_internet=use_internet,
    )


def _push(
    body: str,
    slug: str = "shopee-product-matching-script",
    id_no: int = 16571372,
    title: str = "shopee product matching script",
    is_public: bool = False,
    disable_gpu: bool = False,
    disable_internet: bool = False,
    dataset_source: Optional[List[str]] = None,
    kernel_source: Optional[List[str]] = None,
) -> KernelPushResponse:
    dataset_sources = [] if dataset_source is None else list(dataset_source)
    kernel_sources = [] if kernel_source is None else list(kernel_source)
    competition_sources = ["shopee-product-matching"]
    dataset_sources.append("ar90ngas/shopeeproductmatchingrequirements")
    dataset_sources.append("ar90ngas/shopeeproductmatchingrequirements2")
    dataset_sources.append("ar90ngas/timm-pretrained-efficientnet")
    dataset_sources.append("ar90ngas/timm-pretrained-nfnet")

    return kaggle.push(
        id_no=id_no,
        slug=slug,
        title=title,
        body=body,
        is_public=is_public,
        disable_gpu=disable_gpu,
        disable_internet=disable_internet,
        dataset_sources=dataset_sources,
        kernel_sources=kernel_sources,
        competition_sources=competition_sources,
    )


@app.command()
def push(
    ctx: typer.Context,
    code_file: Path,
    sweep_id: str,
    disable_gpu: bool = False,
    disable_internet: bool = False,
    strict: bool = False,
    sweep_count: int = 1,
) -> None:
    envs = _parse_extra_args(ctx.args)
    envs["SWEEP_ID"] = sweep_id
    envs["SWEEP_COUNT"] = str(sweep_count)

    body = _build(
        input=code_file,
        envs=envs,
        use_internet=not disable_internet,
        strict=False,
    )
    response = _push(
        body=body,
        disable_gpu=disable_gpu,
        disable_internet=disable_internet,
    )
    kaggle.print_response(response)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def submit(
    ctx: typer.Context,
    input: Path,
    strict: bool = False,
) -> None:
    envs = _parse_extra_args(ctx.args)

    script_kernel = _build(
        input=input,
        prologue="",
        secret_keys=[],
        envs=envs,
        strict=strict,
        use_internet=False,
    )

    response = _push(
        body=script_kernel,
        disable_internet=True,
    )
    kaggle.print_response(response)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
