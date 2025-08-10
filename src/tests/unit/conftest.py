# src/tests/unit/conftest.py
import pytest
import pytest_asyncio
import asyncio
from typing import AsyncGenerator
from httpx import AsyncClient, ASGITransport  # Добавить ASGITransport

from src.main import app
from src.infrastructure.mongo import MongoDBManager, set_mongodb_manager


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def mock_mongodb_manager():
    manager = MongoDBManager(
        mongo_uri="mongodb://localhost:27017",
        database_name="test_db",
        is_testing=True,
    )
    await manager.connect()
    set_mongodb_manager(manager)
    yield manager
    await manager.disconnect()


@pytest_asyncio.fixture
async def test_client_mock(mock_mongodb_manager) -> AsyncGenerator[AsyncClient, None]:
    """Тестовый клиент с mock MongoDB"""
    transport = ASGITransport(app=app)  # Создаем transport
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
