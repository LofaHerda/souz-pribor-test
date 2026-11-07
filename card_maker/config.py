"""Конфигурация проекта на базе Pydantic v2 и фабрика создания OpenAI-совместимого клиента."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#лимиты и настройки
NO_DATA = "данных нет"

MAX_TITLE_LEN = 70
MIN_DESC_LEN = 800
MAX_DESC_LEN = 1200
TARGET_DESC_LEN = 1000
MIN_QUERIES = 3
MAX_QUERIES = 5

#параметры работы с LLM API
MAX_ATTEMPTS = 6
LLM_TEMPERATURE = 0.3
REQUEST_TIMEOUT_SEC = 60.0
MAX_SDK_RETRIES = 2

DEFAULT_MODEL = "gpt-4o-mini"


class ConfigError(Exception):
    """Ошибка отсутствия или некорректности настроек в .env."""


def load_env_file() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(override=False)
    for proxy_var in ("ALL_PROXY", "all_proxy"):
        if os.environ.get(proxy_var, "").startswith("socks://"):
            del os.environ[proxy_var]


load_env_file()


class AppConfig(BaseModel):

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        populate_by_name=True,
        extra="ignore",
    )

    api_key: str = Field(min_length=1, alias="OPENAI_API_KEY")
    base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    model: str = Field(default=DEFAULT_MODEL, min_length=1, alias="OPENAI_MODEL")
    project: str | None = Field(default=None, alias="OPENAI_PROJECT")

    timeout: float = REQUEST_TIMEOUT_SEC
    max_retries: int = MAX_SDK_RETRIES
    temperature: float = LLM_TEMPERATURE
    max_attempts: int = MAX_ATTEMPTS

    @classmethod
    def from_env(cls) -> AppConfig:
        non_empty_env = {k: v for k, v in os.environ.items() if v and v.strip()}
        try:
            return cls.model_validate(non_empty_env)
        except ValidationError as exc:
            raise ConfigError(
                "Не найден или пуст OPENAI_API_KEY. Заполните настройки в файле .env"
            ) from exc


class LLMClientFactory:

    @classmethod
    def create(cls, config: AppConfig | None = None) -> tuple[OpenAI, str]:
        cfg = config or AppConfig.from_env()
        client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            project=cfg.project,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
        )
        return client, cfg.model
