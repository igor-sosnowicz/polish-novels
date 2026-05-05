from typing import Literal
import asyncio

import ollama
from pydantic import ValidationError
from tqdm import tqdm

from src.config import Config, get_config
from src.models import InteractionList


MODELS = Literal[
    "SpeakLeash/bielik-1.5b-v3.0-instruct:Q8_0",
    "SpeakLeash/bielik-minitron-7B-v3.0-instruct:Q4_K_M",
]

config: Config = get_config()


class LLMProcessor:
    def __init__(self) -> None:
        self.client = ollama.AsyncClient()
        self.model_name = config.llm_model
        self.max_chunk_size = config.max_chunk_size
        self.system_prompt = config.system_prompt

    async def initialize(self):
        await self._download_model_if_missing()

    async def _download_model_if_missing(self):
        try:
            models_dict: ollama.ListResponse = await self.client.list()
            available_models = [m["model"] for m in models_dict.get("models", [])]

            if not any(self.model_name in name for name in available_models):
                print(f"Model '{self.model_name}' not found. Downloading...")
                await self.client.pull(self.model_name)
                print(f"Model downloaded: {self.model_name}")
            else:
                print(f"Model '{self.model_name}' is ready for use.")
        except Exception as e:
            raise RuntimeError(f"Error occurred while communicating with Ollama: {e}")

    async def unload_model(self):
        print(
            f"\n[Resource Management] Unloading model '{self.model_name}' from VRAM..."
        )
        try:
            await self.client.generate(model=self.model_name, prompt="", keep_alive=0)
            print("[Resource Management] Cleanup completed successfully.")
        except Exception as e:
            print(f"[Resource Management] Error occurred while cleaning up VRAM: {e}")

    def _text_into_chunks(self, text: str) -> list[str]:
        chunks = [chunk for chunk in text.split("\n\n") if chunk]
        final_chunks = []

        for chunk in chunks:
            if len(chunk) < self.max_chunk_size:
                final_chunks.append(chunk)
                continue

            sub_chunks = []
            current = ""
            for word in chunk.split():
                if len(current + " " + word) > self.max_chunk_size:
                    if current:
                        sub_chunks.append(current)
                        current = word
                    else:
                        sub_chunks.append(word)
                        current = ""
                else:
                    current += " " + word if current else word
            if current:
                sub_chunks.append(current)
            final_chunks.extend(sub_chunks)

        return final_chunks

    async def _process_chunk(
        self, text: str, previous_characters: set[str] | None = None
    ) -> InteractionList:
        previous_characters = previous_characters if previous_characters else set()

        context_addon: str = ""
        if previous_characters:
            chars_str = ", ".join(sorted(previous_characters))
            context_addon = (
                f"\n\nW poprzednim fragmencie zostały znalezione postacie: {chars_str}."
            )

        prompt = "{system_prompt}{context_addon}\n\nFragment:\n{text}".format(
            system_prompt=self.system_prompt, context_addon=context_addon, text=text
        )

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                response = await self.client.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    format=InteractionList.model_json_schema(),
                    options=config.ollama_options,
                )
                json_response = response["message"]["content"]
                return InteractionList.model_validate_json(json_response)

            except ValidationError as e:
                print(f"\nValidation Error (chunk {attempt}/{max_attempts}): {e}")
            except Exception as e:
                print(f"\nChunk processing error (chunk {attempt}/{max_attempts}): {e}")

            if attempt < max_attempts:
                await asyncio.sleep(1)

        print("\nFailed to process chunk after multiple attempts. Skipping this chunk.")
        return InteractionList(interactions=[])

    async def process_book(self, book_text: str) -> InteractionList:
        interactions = InteractionList(interactions=[])
        chunks = self._text_into_chunks(book_text)
        previous_characters = set()

        for chunk in tqdm(chunks, desc="Przetwarzanie fragmentów", unit="chunk"):
            chunk_result = await self._process_chunk(chunk, previous_characters)
            interactions += chunk_result

            previous_characters = set()
            for interaction in chunk_result.interactions:
                previous_characters.add(interaction.character_1)
                previous_characters.add(interaction.character_2)

        return interactions
