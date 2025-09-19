# src/datasets_api/service.py
from typing import Optional

import aiohttp
from aiohttp import TCPConnector, ClientTimeout

from src.domain.services.llm_dto import LLMQuestion, LLMQuestionWithDatasets
from src.infrastructure.config import LLM_URL, LLM_OPEN_AI_KEY
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

    async def openai_by_api(
        self, system_prompt: str, messages: list[str] | None = None
    ):
        if messages is None:
            messages = []

        url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {LLM_OPEN_AI_KEY}",
            "Content-Type": "application/json",
        }

        data = {
            "model": "gpt-5-nano",
            "messages": (
                [{"role": "system", "content": system_prompt}]
                + [{"role": "user", "content": m} for m in messages]
            ),
            # "max_completion_tokens": max_completion_tokens,
        }

        async with self.session.post(url, json=data, headers=headers) as response:
            if response.status == 200:
                response_json = await response.json()
                return response_json["choices"][0]["message"]["content"]
            else:
                error = await response.text()
                raise Exception(f"Error {response.status}: {error}")

    # endregion

    # region REQUESTS

    system_prompt = """You are an urban data researcher specializing in city analytics and evidence-based urban planning. Your task is to provide data-driven insights based on available datasets from Chemnitz, Saxony, Germany. Always ground your analysis in the specific data provided and clearly reference which datasets support your conclusions."""

    async def get_research_questions(self, initial_question: str) -> list[LLMQuestion]:
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
        result = await self.openai_by_api(
            system_prompt=self.system_prompt, messages=[prompt]
        )
        try:
            valid_result = extract_json(result)
            return [LLMQuestion(**item) for item in valid_result]
        except Exception as e:
            logger.error(f"Failed to parse research questions: {e}", exc_info=True)
            raise

    async def answer_research_question(
        self, question: LLMQuestionWithDatasets
    ) -> list[LLMQuestion]:
        context = question.to_llm_context()
        instructions = """
            ## Analysis Guidelines

            1. **Data-Driven Approach**: Base all insights on the specific fields and statistics provided in the datasets above
            2. **Field Utilization**: Leverage the statistical summaries, distributions, and temporal ranges to provide quantitative insights
            3. **Cross-Dataset Analysis**: When multiple datasets are relevant, identify potential relationships between their fields
            4. **Statistical Rigor**: Use the provided metrics (mean, median, quartiles, standard deviation) to describe patterns and outliers
            5. **Temporal Context**: Consider the creation and modification dates of datasets when interpreting their relevance
            6. **Data Quality**: Account for null values and unique counts when assessing data completeness and reliability
            7. **Urban Context**: Frame insights within the specific context of Chemnitz as a mid-sized German city in Saxony

            ## Response Requirements

            - Cite specific dataset names and field names when making claims
            - Quantify findings using the provided statistical measures
            - Acknowledge data limitations based on null counts and temporal coverage
            - Provide actionable insights relevant to urban planning and city management
            - Structure the response with clear sections for different aspects of the analysis
            - Only 1 paragraph of text, be laconic
            - If datasets aren't suitable, just tell 'No suitable datasets', do not make assumptions or suggestions
            """
        prompt = f"""{context}\n{instructions}"""

        try:
            result = await self.openai_by_api(
                system_prompt=self.system_prompt, messages=[prompt]
            )
            return result
        except Exception as e:
            logger.error(f"Failed to answer the research question: {e}", exc_info=True)
            raise

    async def summary(self, messages: list[str]):
        prompt = "Summarize those paragraphs. Use 1 paragraph in answer"

        try:
            result = await self.openai_by_api(
                system_prompt=self.system_prompt, messages=messages + [prompt]
            )
            return result
        except Exception as e:
            logger.error(f"Failed to answer the research question: {e}", exc_info=True)
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
