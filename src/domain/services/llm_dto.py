import hashlib
import json
from dataclasses import dataclass, asdict

import numpy as np

from src.datasets.datasets_metadata import FieldNumeric, FieldDate, FieldString
from src.datasets_api.datasets_dto import DatasetResponse


@dataclass
class LLMQuestion:
    question: str
    reason: str

    @property
    def question_hash(self) -> str:
        return hashlib.sha256(self.question.encode()).hexdigest()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["question_hash"] = self.question_hash
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class LLMQuestionWithEmbeddings(LLMQuestion):
    embeddings: np.ndarray

    def to_json(self) -> str:
        data = asdict(self)
        data["question_hash"] = self.question_hash
        if isinstance(self.embeddings, np.ndarray):
            data["embeddings"] = self.embeddings.tolist()
        else:
            data["embeddings"] = self.embeddings
        return json.dumps(data)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["question_hash"] = self.question_hash
        if isinstance(self.embeddings, np.ndarray):
            data["embeddings"] = self.embeddings.tolist()
        else:
            data["embeddings"] = self.embeddings
        return data


@dataclass
class LLMQuestionWithDatasets(LLMQuestion):
    datasets: list[DatasetResponse]

    def to_json(self) -> str:
        data = asdict(self)
        data["question_hash"] = self.question_hash
        data["datasets"] = [ds.to_dict() for ds in self.datasets]
        data.pop("reason", None)
        return json.dumps(data)

    def to_llm_context(self) -> str:
        """Generate comprehensive context for LLM from LLMQuestionWithDatasets instance"""

        context_parts: list[str] = []

        # Question section
        context_parts.append(f"\n## Research Question\n{self.question}")

        # Reason for question
        context_parts.append(f"\n## Motivation for the question\n{self.reason}")

        # Datasets section
        context_parts.append(f"\n## Available Datasets ({len(self.datasets)} datasets)")

        for i, dataset_response in enumerate(self.datasets, 1):
            ds = dataset_response.metadata
            context_parts.append(f"\n### Dataset {i}: {ds.title}")
            context_parts.append(f"**Relevance Score**: {dataset_response.score:.2f}")

            # Basic metadata
            if ds.description:
                context_parts.append(f"**Description**: {ds.description}")
            if ds.organization:
                context_parts.append(f"**Organization**: {ds.organization}")
            if ds.url:
                context_parts.append(f"**Source URL**: {ds.url}")
            if ds.author:
                context_parts.append(f"**Author**: {ds.author}")

            # Location information
            location_parts = []
            if ds.city:
                location_parts.append(ds.city)
            if ds.state:
                location_parts.append(ds.state)
            if ds.country:
                location_parts.append(ds.country)
            if location_parts:
                context_parts.append(f"**Location**: {', '.join(location_parts)}")

            # Temporal information
            if ds.metadata_created:
                context_parts.append(f"**Created**: {ds.metadata_created}")
            if ds.metadata_modified:
                context_parts.append(f"**Last Modified**: {ds.metadata_modified}")

            # Tags and groups
            if ds.tags:
                context_parts.append(f"**Tags**: {', '.join(ds.tags)}")
            if ds.groups:
                context_parts.append(f"**Groups**: {', '.join(ds.groups)}")

            # Fields information - CRITICAL DATA
            if ds.fields:
                context_parts.append(f"\n**Dataset Fields** ({len(ds.fields)} fields):")

                for field_name, field_info in ds.fields.items():
                    context_parts.append(f"\n**Field: `{field_name}`**")
                    context_parts.append(f"- Type: {field_info.type}")
                    context_parts.append(f"- Unique values: {field_info.unique_count}")
                    context_parts.append(f"- Null values: {field_info.null_count}")

                    # Numeric field statistics
                    if isinstance(field_info, FieldNumeric):
                        context_parts.append("- Statistical Summary:")
                        context_parts.append(f"  - Mean: {field_info.mean:.2f}")
                        context_parts.append(f"  - Std Dev: {field_info.std:.2f}")
                        context_parts.append(
                            f"  - Min: {field_info.quantile_0_min:.2f}"
                        )
                        context_parts.append(
                            f"  - Q1 (25%): {field_info.quantile_25:.2f}"
                        )
                        context_parts.append(
                            f"  - Median (50%): {field_info.quantile_50_median:.2f}"
                        )
                        context_parts.append(
                            f"  - Q3 (75%): {field_info.quantile_75:.2f}"
                        )
                        context_parts.append(
                            f"  - Max: {field_info.quantile_100_max:.2f}"
                        )
                        context_parts.append(
                            f"  - Distribution: {field_info.distribution}"
                        )

                    # Date field statistics
                    elif isinstance(field_info, FieldDate):
                        context_parts.append("- Temporal Range:")
                        context_parts.append(
                            f"  - Earliest: {field_info.min.isoformat()}"
                        )
                        context_parts.append(
                            f"  - Latest: {field_info.max.isoformat()}"
                        )
                        context_parts.append(
                            f"  - Mean date: {field_info.mean.isoformat()}"
                        )

                    # String field (no additional statistics)
                    elif isinstance(field_info, FieldString):
                        pass  # Basic info already included

            context_parts.append("")  # Empty line between datasets

        return "\n".join(context_parts)
