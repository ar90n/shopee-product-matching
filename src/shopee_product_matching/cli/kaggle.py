from functools import singledispatch
from typing import Any, Iterable

import typer
from kaggle import KaggleApi
from kaggle.api_client import ApiClient
from kaggle.models.kaggle_models_extended import (
    DatasetNewResponse,
    DatasetNewVersionResponse,
    KernelPushResponse,
)
from kaggle.models.kernel_push_request import KernelPushRequest


def get_kaggle_api() -> KaggleApi:
    api = KaggleApi(ApiClient())
    api.authenticate()
    return api


@singledispatch
def print_response(result: Any) -> None:
    raise NotImplementedError(f"print_response not support {type(result)}")


@print_response.register
def _print_KernelPushResponse(result: KernelPushResponse) -> None:
    if result.error is not None:
        typer.echo("error: {}".format(result.error))
    else:
        typer.echo("ref: {}".format(result.ref))
        typer.echo("url: {}".format(result.url))
        typer.echo("version: {}".format(result.versionNumber))


@print_response.register
def _print_DatasetNewResponse(result: DatasetNewResponse) -> None:
    if result.status == "error":
        typer.echo("error: {}".format(result.error))
    else:
        typer.echo("ref: {}".format(result.ref))
        typer.echo("url: {}".format(result.url))


@print_response.register
def _print_DatasetNewVersionResponse(result: DatasetNewVersionResponse) -> None:
    if result.status == "error":
        typer.echo("error: {}".format(result.error))
    else:
        typer.echo("ref: {}".format(result.ref))
        typer.echo("url: {}".format(result.url))


def push(
    id_no: int,
    slug: str,
    title: str,
    body: str,
    is_public: bool,
    disable_gpu: bool,
    disable_internet: bool,
    dataset_sources: Iterable[str],
    kernel_sources: Iterable[str],
    competition_sources: Iterable[str],
) -> KernelPushResponse:
    kernel_push_request = KernelPushRequest(
        id=id_no,
        slug=slug,
        new_title=title,
        text=body,
        language="python",
        kernel_type="script",
        is_private=not is_public,
        enable_gpu=not disable_gpu,
        enable_internet=not disable_internet,
        dataset_data_sources=list(dataset_sources),
        kernel_data_sources=list(kernel_sources),
        competition_data_sources=list(competition_sources),
        category_ids=[],
    )

    api = get_kaggle_api()
    return KernelPushResponse(
        api.process_response(
            api.kernel_push_with_http_info(kernel_push_request=kernel_push_request)
        )
    )
