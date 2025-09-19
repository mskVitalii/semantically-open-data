import json
import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.responses import StreamingResponse

from .datasets_dto import DatasetSearchRequest, DatasetSearchResponse
from ..domain.services.dataset_service import DatasetService, get_dataset_service
from ..domain.services.llm_dto import (
    LLMQuestion,
    LLMQuestionWithEmbeddings,
    LLMQuestionWithDatasets,
)
from ..domain.services.llm_service import (
    LLMService,
    get_llm_service_dep,
)
from ..infrastructure.logger import get_prefixed_logger
from ..vector_search.embedder import embed_batch_with_ids

logger = get_prefixed_logger("API /datasets")

router = APIRouter(prefix="/datasets", tags=["datasets"])


# region Search Datasets
@router.post("/search_datasets", response_model=DatasetSearchResponse)
async def search_datasets(
    request: DatasetSearchRequest,
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetSearchResponse:
    """
    Dataset search

    Supported parameters:
    - query: full-text search by name and description
    - tags: filter by tags
    - limit: number of results (1–100)
    - offset: pagination offset
    """
    try:
        return await service.search_datasets(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# endregion


# region Bootstrap
@router.post("/bootstrap")
async def bootstrap(
    service: DatasetService = Depends(get_dataset_service),
):
    try:
        result = await service.bootstrap_datasets()
        return {"ok": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# endregion


# region QA
async def step_0_llm_questions(
    step: int, question: str, llm_service: LLMService
) -> list[LLMQuestion]:
    logger.info(f"step: {step}. LLM QUESTIONS start")
    start_0 = time.perf_counter()

    research_questions = await llm_service.get_research_questions(question)
    elapsed_0 = time.perf_counter() - start_0
    logger.info(f"step: {step}. LLM QUESTIONS end (elapsed: {elapsed_0:.2f}s)")
    logger.info(research_questions)
    return research_questions


async def step_1_embeddings(
    step: int, research_questions: list[LLMQuestion]
) -> list[LLMQuestionWithEmbeddings]:
    logger.info(f"step: {step}. EMBEDDINGS start")
    start_1 = time.perf_counter()

    questions_list = [
        {"text": q.question, "id": q.question_hash} for q in research_questions
    ]
    embeddings = await embed_batch_with_ids(questions_list)

    embeddings_map: dict[str, np.ndarray] = {
        str(e["id"]): e["embedding"] for e in embeddings
    }

    questions_with_embeddings: list[LLMQuestionWithEmbeddings] = [
        LLMQuestionWithEmbeddings(
            question=q.question,
            reason=q.reason,
            embeddings=embeddings_map.get(q.question_hash),
        )
        for q in research_questions
    ]

    elapsed_1 = time.perf_counter() - start_1
    logger.info(f"step: {step}. EMBEDDINGS end (elapsed: {elapsed_1:.2f}s)")
    return questions_with_embeddings


async def generate_events(
    question: str, datasets_service: DatasetService, llm_service: LLMService
):
    step = 0
    try:
        # region 0. LLM QUESTIONS
        # research_questions: list[LLMQuestion] | None = None
        # if not IS_DOCKER:
        #     cached_questions = await check_qa_cache(question, step)
        #     if cached_questions is not None:
        #         research_questions = [
        #             LLMQuestion(cq["question"], reason=cq["reason"])
        #             for cq in cached_questions
        #         ]
        # if research_questions is None:
        research_questions = await step_0_llm_questions(step, question, llm_service)
        yield f"data: {
            json.dumps(
                {
                    'step': step,
                    'status': 'OK',
                    'data': {
                        'question': question,
                        'research_question': [q.to_json() for q in research_questions],
                    },
                }
            )
        }\n\n"
        # if not IS_DOCKER:
        #     await set_qa_cache(
        #         question, step, [q.to_dict() for q in research_questions]
        #     )
        step += 1
        # endregion

        # mb: make class Question & correct types
        # mb, return many mini-steps for each query
        # mb I should provide IDs within the full system to keep the order & do not mix questions embeddings

        # region 1. EMBEDDINGS
        embeddings = await step_1_embeddings(step, research_questions)
        yield f"data: {
            json.dumps(
                {
                    'step': step,
                    'status': 'OK',
                    'data': [e.to_dict() for e in embeddings],
                }
            )
        }\n\n"
        step += 1
        # endregion

        # region 2. VECTOR SEARCH
        logger.info(f"step: {step}. VECTOR SEARCH start")
        start_2 = time.perf_counter()

        result_questions_with_datasets: list[LLMQuestionWithDatasets] = []
        for i, embedding in enumerate(embeddings):
            datasets = await datasets_service.search_datasets_with_embeddings(
                embedding.embeddings
            )
            result_questions_with_datasets.append(
                LLMQuestionWithDatasets(
                    question=embedding.question,
                    reason=embedding.reason,
                    datasets=datasets,
                )
            )
            yield f"data: {
                json.dumps(
                    {
                        'step': step,
                        'sub_step': i,
                        'status': 'OK',
                        'data': {
                            'hash': embedding.question_hash,
                            'datasets': [ds.to_dict() for ds in datasets],
                        },
                    }
                )
            }\n\n"
        # logger.info(result_questions_with_datasets)
        elapsed_2 = time.perf_counter() - start_2
        logger.info(f"step: {step}. VECTOR SEARCH end (elapsed: {elapsed_2:.2f}s)")
        step += 1
        # endregion

        # mb choose fields to reduce context and improve results

        # region 3. INTERPRETATION
        logger.info(f"step: {step}. INTERPRETATION start")
        start_3 = time.perf_counter()

        for i, q in enumerate(result_questions_with_datasets):
            start_3_i = time.perf_counter()
            logger.info(f"step: {step}.{str(i)} INTERPRETATION STEP start")
            answer = await llm_service.answer_research_question(q)
            logger.info(answer)
            yield f"data: {
                json.dumps(
                    {
                        'step': step,
                        'sub_step': i,
                        'status': 'OK',
                        'data': {
                            'question_hash': q.question_hash,
                            'answer': answer,
                        },
                    }
                )
            }\n\n"
            elapsed_3_i = time.perf_counter() - start_3_i
            logger.info(
                f"step: {step}.{str(i)} INTERPRETATION STEP end (elapsed: {elapsed_3_i:.2f}s)"
            )
        elapsed_3 = time.perf_counter() - start_3
        logger.info(f"step: {step}. INTERPRETATION end (elapsed: {elapsed_3:.2f}s)")
        step += 1
        # endregion

        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error("ERROR!", exc_info=True)
        yield f"data: {json.dumps({'step': step, 'status': 'error', 'error': str(e)})}\n\n"


@router.get("/qa")
async def stream(
    question: str = Query(
        "What is the color of grass in Germany?", description="Ask the system"
    ),
    datasets_service: DatasetService = Depends(get_dataset_service),
    llm_service: LLMService = Depends(get_llm_service_dep),
):
    return StreamingResponse(
        generate_events(question, datasets_service, llm_service),
        media_type="text/event-stream",
    )


# endregion
