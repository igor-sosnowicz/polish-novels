"""Module for fetching data via the API of wolnelektury.pl."""

import httpx
from diskcache import Cache

cache = Cache(".cache")


async def fetch_epochs() -> list[str]:
    if "epochs" in cache:
        return cache["epochs"]
    excluded = {"nie dotyczy"}
    url = "https://wolnelektury.pl/api/epochs/"
    async with httpx.AsyncClient() as client:
        response = await client.get(url=url)
        response.raise_for_status()
        epochs = [
            epoch["slug"] for epoch in response.json() if epoch["name"] not in excluded
        ]
        cache["epochs"] = epochs
        return epochs


async def check_txt_exists(client: httpx.AsyncClient, book_href: str) -> bool:
    cache_key = f"has_txt_{book_href}"
    if cache_key in cache:
        return cache[cache_key]

    try:
        response = await client.get(book_href)
        response.raise_for_status()
        has_txt = "txt" in response.json()
        cache[cache_key] = has_txt
        return has_txt
    except Exception:
        return False


async def fetch_prose(limit: int, epochs: list[str]) -> dict[str, list[str]]:
    kind = "epika"
    cache_key = f"books_kind_{kind}"

    if cache_key in cache:
        all_books = cache[cache_key]
    else:
        url = f"https://wolnelektury.pl/api/kinds/{kind}/books/"
        async with httpx.AsyncClient() as client:
            response = await client.get(url=url, timeout=45)
            response.raise_for_status()
            all_books = response.json()
            cache[cache_key] = all_books

    result = {}
    async with httpx.AsyncClient() as client:
        for epoch_slug in epochs:
            epoch_books = [
                b for b in all_books if b["epoch"].lower() == epoch_slug.lower()
            ]

            valid_hrefs = []
            # We check books in batches to find enough with TXT versions
            for book in epoch_books:
                if len(valid_hrefs) >= limit:
                    break
                if await check_txt_exists(client, book["href"]):
                    valid_hrefs.append(book["href"])

            result[epoch_slug] = valid_hrefs

    return result


async def fetch_book(url: str) -> str | None:
    if url in cache:
        return cache[url]
    async with httpx.AsyncClient() as client:
        response = await client.get(url=url)
        response.raise_for_status()
        data = response.json()
        txt_url = data.get("txt")
        if not txt_url:
            return None
        txt_response = await client.get(url=txt_url)
        txt_response.raise_for_status()
        text = txt_response.text
        cache[url] = text
        return text


def to_filename(url: str) -> str:
    filename = url.removeprefix("https://wolnelektury.pl/api/books/")
    filename = filename.removesuffix("/")
    filename = filename.replace("-", "_")
    return f"{filename}.txt"
