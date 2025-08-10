# src/tests/unit/test_health.py
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_root_endpoint(test_client_mock: AsyncClient):
    response = await test_client_mock.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Semantic Open Data API is running"}


@pytest.mark.asyncio
async def test_health_check(test_client_mock: AsyncClient):
    response = await test_client_mock.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
