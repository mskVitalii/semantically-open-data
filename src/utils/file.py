from pathlib import Path

import aiofiles
from typing import Union

import asyncio


async def save_file(path: str, content: Union[str, bytes], binary: bool = False):
    if binary:
        async with aiofiles.open(path, "wb") as f:
            await f.write(content)
    else:
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(content)


def save_file_with_task(
    path: str | Path, content: Union[str, bytes], binary: bool = False
):
    asyncio.create_task(save_file(path, content, binary))
