import json
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.responses import StreamingResponse

from .datasets_dto import DatasetSearchRequest, DatasetSearchResponse
from ..domain.services.dataset_service import DatasetService, get_dataset_service
from ..domain.services.llm_service import (
    LLMService,
    get_llm_service_dep,
)
from ..infrastructure.logger import get_prefixed_logger
from ..vector_search.embedder import embed_batch

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
async def generate_events(
    question: str, datasets_service: DatasetService, llm_service: LLMService
):
    step = 0
    try:
        # region 0. LLM QUESTIONS
        logger.info(f"step: {step}. LLM QUESTIONS start")
        start_0 = time.perf_counter()

        research_questions = await llm_service.get_research_questions(question)
        research_questions_with_id = [
            {**q, "id": i} for i, q in enumerate(research_questions)
        ]
        elapsed_0 = time.perf_counter() - start_0
        logger.info(f"step: {step}. LLM QUESTIONS end (elapsed: {elapsed_0:.2f}s)")
        logger.info(research_questions_with_id)
        yield f"data: {
            json.dumps(
                {
                    'step': step,
                    'status': 'OK',
                    'data': {
                        'question': question,
                        'research_question': research_questions_with_id,
                    },
                }
            )
        }\n\n"
        step += 1
        # endregion

        # mb: make class Question & correct types
        # mb, return many mini-steps for each query
        # mb I should provide IDs within the full system to keep the order & do not mix questions embeddings

        # region 1. EMBEDDINGS
        logger.info(f"step: {step}. EMBEDDINGS start")
        start_1 = time.perf_counter()

        questions_list = [q.get("question", "") for q in research_questions_with_id]
        embeddings = await embed_batch(questions_list)
        embeddings_with_id_to_list = [
            {"embeddings": q.tolist(), "id": i} for i, q in enumerate(embeddings)
        ]
        elapsed_1 = time.perf_counter() - start_1
        logger.info(f"step: {step}. EMBEDDINGS end (elapsed: {elapsed_1:.2f}s)")

        yield f"data: {
            json.dumps(
                {
                    'step': step,
                    'status': 'OK',
                    'data': embeddings_with_id_to_list,
                }
            )
        }\n\n"

        step += 1
        # endregion

        # region 2. VECTOR SEARCH
        logger.info(f"step: {step}. VECTOR SEARCH start")
        start_2 = time.perf_counter()

        embeddings_with_id = [
            {"embeddings": q, "id": i} for i, q in enumerate(embeddings)
        ]
        for i, embedding in enumerate(embeddings_with_id):
            datasets = await datasets_service.search_datasets_with_embeddings(
                embedding.get("embeddings")
            )
            yield f"data: {
                json.dumps(
                    {
                        'step': step,
                        'sub_step': i,
                        'status': 'OK',
                        'data': {'id': i, 'datasets': datasets},
                    }
                )
            }\n\n"

        elapsed_2 = time.perf_counter() - start_2
        logger.info(f"step: {step}. VECTOR SEARCH end (elapsed: {elapsed_2:.2f}s)")
        step += 1
        # endregion

        # region 3. MONGO REQUESTS & RESPONSES

        # 3.1 определить для каждого запроса, что может быть - среднее / сумма

        # endregion

        # region 3. INTERPRETATION

        # 3.1 определить для каждого запроса, что может быть - среднее / сумма

        # endregion

        # TODO: ask LLM for each to create short paragraph. Include link to the datasets
        #   yield интерпретацию всех этих данных нейронкой
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'step': step, 'status': 'error', 'error': str(e)})}\n\n"


@router.get("/qa")
async def stream(
    question: str = Query(None, description="Ask the system"),
    datasets_service: DatasetService = Depends(get_dataset_service),
    llm_service: LLMService = Depends(get_llm_service_dep),
):
    return StreamingResponse(
        generate_events(question, datasets_service, llm_service),
        media_type="text/event-stream",
    )


# endregion
