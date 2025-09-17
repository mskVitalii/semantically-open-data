# src/datasets_api/service.py
from typing import Optional

import aiohttp
from aiohttp import TCPConnector, ClientTimeout

from src.domain.services.llm_dto import LLMQuestion
from src.infrastructure.config import LLM_URL
from src.infrastructure.logger import get_prefixed_logger
from src.utils.llm_utils import extract_json

logger = get_prefixed_logger(__name__, "LLM_SERVICE")


class LLMService:
    """Service for working with LLM"""

    # region INIT

    def __init__(self, base_url: str):
        self.base_url = base_url
        # Create connector with connection pooling
        connector = TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,  # DNS cache for 5 minutes
            enable_cleanup_closed=True,
            force_close=True,
        )

        # Optimized timeout settings
        timeout = ClientTimeout(
            total=1200,  # Total timeout
            connect=10,  # Connection timeout
            sock_read=1190,  # Socket read timeout
        )

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": "LLM Asker (Python/aiohttp)",
                "Accept-Encoding": "gzip, deflate",  # Enable compression
            },
        )

    async def close_llm_session(self):
        logger.info("LLMService >> __aexit__")
        if self.session:
            await self.session.close()

    # endregion

    # region LOGIC

    async def ollama_by_api(self, prompt: str, temperature=0.5, max_tokens=10):
        url = self.base_url + "/api/generate"

        data = {
            "model": "gemma3:4b",
            "prompt": prompt,
            "options": {"temperature": temperature, "max_tokens": max_tokens},
            "stream": False,
        }

        async with self.session.post(url, json=data) as response:
            if response.status == 200:
                response_json = await response.json()
                return extract_json(response_json["response"])
            else:
                error = await response.text()
                raise Exception(f"Error {response.status}: {error}")

    # endregion

    # region REQUESTS

    async def get_research_questions(self, initial_question) -> list[LLMQuestion]:
        prompt = f"""
        Question: {initial_question}.
        Generate N research questions you would need to know to answer the question.

        Output **only** valid JSON, nothing else.
        Do not include explanations, markdown, code fences or text outside JSON.
        The output must be an array of objects with fields "question" and "reason".

        Example:
        [
          {{"question": "What is X?", "reason": "Because ..."}},
          {{"question": "How does Y work?", "reason": "Because ..."}}
        ]
        """
        result = await self.ollama_by_api(prompt)
        try:
            return [LLMQuestion(**item) for item in result]
        except Exception as e:
            logger.error(f"Failed to parse research questions: {e}", exc_info=True)
            raise

    # endregion


# region DI
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get MongoDB manager instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(base_url=LLM_URL)
    return _llm_service


async def get_llm_service_dep() -> LLMService:
    """Dependency to get database"""
    return get_llm_service()


# endregion
