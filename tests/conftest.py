import pytest

from src.checks import physics_checks
from src.config import CONFIG
from src.experiments import random_streams


@pytest.fixture(scope="session")
def physics() -> dict[str, object]:
    return physics_checks(CONFIG, random_streams(CONFIG.seed)["checks"])
