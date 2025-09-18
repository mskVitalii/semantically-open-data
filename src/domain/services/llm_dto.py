import hashlib
import json
from dataclasses import dataclass, asdict

import numpy as np

from src.datasets_api.datasets_dto import DatasetResponse


@dataclass
class LLMQuestion:
    question: str
    reason: str

    @property
    def question_hash(self) -> str:
        return hashlib.sha256(self.question.encode()).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict() | {"question_hash": self.question_hash})


@dataclass
class LLMQuestionWithEmbeddings(LLMQuestion):
    embeddings: np.ndarray

    def to_json(self) -> str:
        data = asdict(self)
        data["question_hash"] = self.question_hash
        data["embeddings"] = [e.tolist() for e in self.embeddings]
        data.pop("reason", None)
        return json.dumps(data)


@dataclass
class LLMQuestionWithDatasets(LLMQuestion):
    datasets: list[DatasetResponse]

    def to_json(self) -> str:
        data = asdict(self)
        data["question_hash"] = self.question_hash
        data["datasets"] = [ds.to_dict() for ds in self.datasets]
        data.pop("reason", None)
        return json.dumps(data)
