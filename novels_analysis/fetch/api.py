"""Module for fetching data via the API of wolnelektury.pl."""

import asyncio
import httpx
from diskcache import Cache
from loguru import logger
from typing import cast

from novels_analysis.config.configuration import MIN_BOOK_LENGTH

cache = Cache(".cache")
Book = dict[str, str]
DEFAULT_TIMEOUT = httpx.Timeout(45.0)
MAX_RETRIES = 3


async def _request_with_retries(
    client: httpx.AsyncClient,
    url: str,
    timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
) -> httpx.Response | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await client.get(url=url, timeout=timeout)
            response.raise_for_status()
            return response
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt == MAX_RETRIES:
                logger.warning(
                    f"Request failed after {MAX_RETRIES} attempts for {url}: {exc}."
                )
                return None
            await asyncio.sleep(attempt)
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            logger.warning(
                f"Request returned HTTP {status_code} for {url}. Skipping..."
            )
            return None

    return None


async def fetch_epochs() -> list[str]:
    if "epochs" in cache:
        return cast(list[str], cache["epochs"])
    excluded = {"nie dotyczy"}
    url = "https://wolnelektury.pl/api/epochs/"
    async with httpx.AsyncClient() as client:
        response = await _request_with_retries(client, url=url)
        if response is None:
            raise RuntimeError("Could not fetch epochs from the API.")
        epochs = cast(
            list[str],
            [
                epoch["slug"]
                for epoch in response.json()
                if epoch["name"] not in excluded
            ],
        )
        cache["epochs"] = epochs
        return epochs


async def check_txt_exists(client: httpx.AsyncClient, book_href: str) -> bool:
    cache_key = f"has_txt_{book_href}"
    if cache_key in cache:
        return cast(bool, cache[cache_key])

    try:
        response = await _request_with_retries(client, url=book_href)
        if response is None:
            return False
        has_txt = "txt" in response.json()
        cache[cache_key] = has_txt
        return has_txt
    except Exception:
        return False


async def fetch_prose(epochs: list[str]) -> dict[str, list[str]]:
    kind = "epika"
    cache_key = f"books_kind_{kind}"

    if cache_key in cache:
        all_books = cast(list[Book], cache[cache_key])
    else:
        url = f"https://wolnelektury.pl/api/kinds/{kind}/books/"
        async with httpx.AsyncClient() as client:
            response = await _request_with_retries(client, url=url)
            if response is None:
                raise RuntimeError("Could not fetch books list from the API.")
            all_books = cast(list[Book], response.json())
            cache[cache_key] = all_books

    result: dict[str, list[str]] = {}
    async with httpx.AsyncClient() as client:
        for epoch_slug in epochs:
            epoch_books = [
                b for b in all_books if b["epoch"].lower() == epoch_slug.lower()
            ]

            valid_hrefs = []
            for book in epoch_books:
                if await check_txt_exists(client, book["href"]):
                    valid_hrefs.append(book["href"])

            result[epoch_slug] = valid_hrefs

    return result


async def fetch_book(url: str) -> str | None:
    if url in cache:
        return cast(str, cache[url])
    async with httpx.AsyncClient() as client:
        response = await _request_with_retries(client, url=url)
        if response is None:
            logger.warning(
                f"Could not fetch book metadata for url = {url}. Skipping..."
            )
            return None
        data = cast(dict[str, str], response.json())
        txt_url = data.get("txt")
        if not txt_url:
            logger.warning(
                f"There is no TXT version of the book with url = {url}. Skipping..."
            )
            return None
        txt_response = await _request_with_retries(client, url=txt_url)
        if txt_response is None:
            logger.warning(f"Could not download TXT for url = {url}. Skipping...")
            return None
        text = txt_response.text
        if len(text) < MIN_BOOK_LENGTH:
            logger.warning("The book is too short. Skipping...")
            return None
        cache[url] = text
        logger.success("Fetched a book.")
        return text


def to_filename(url: str) -> str:
    filename = url.removeprefix("https://wolnelektury.pl/api/books/")
    filename = filename.removesuffix("/")
    filename = filename.replace("-", "_")
    return f"{filename}.txt"
