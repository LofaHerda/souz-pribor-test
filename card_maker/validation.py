"""Декларативная валидация входного JSON товара и ответа LLM на базе Pydantic v2."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, field_validator

from .config import MAX_DESC_LEN, MAX_QUERIES, MAX_TITLE_LEN, MIN_DESC_LEN, MIN_QUERIES, NO_DATA

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

_INJECTION_RE = re.compile(
    r"(?:внимание\s+системе|игнорируй\s+предыдущие|ignore\s+previous|system\s+prompt)[^.!?]*[.!?]?",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


class InputError(Exception):
    """Ошибка чтения файла или некорректной структуры входного JSON."""


class ProductSchema(BaseModel):
    """Декларативная схема входного JSON товара (заменяет ручные if-проверки типов)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: NonEmptyStr
    manufacturer: str = ""
    specs: dict[str, str] = Field(default_factory=dict)
    description: str = ""

    @field_validator("manufacturer", "description", mode="before")
    @classmethod
    def _none_to_empty(cls, val: Any) -> str:
        return "" if val is None else str(val)

    @field_validator("specs", mode="before")
    @classmethod
    def _normalize_specs(cls, val: Any) -> dict[str, str]:
        if val is None:
            return {}
        if not isinstance(val, dict):
            raise ValueError("должно быть JSON-объектом {характеристика: значение}")
        return {
            str(k).strip(): ("" if v is None or isinstance(v, (dict, list)) else str(v).strip())
            for k, v in val.items()
            if str(k).strip()
        }


class CommercialBlockSchema(BaseModel):
    """Схема коммерческого блока карточки."""

    suitable_for: list[NonEmptyStr] = Field(min_length=1)
    tasks: list[NonEmptyStr] = Field(min_length=1)


class DraftCardSchema(BaseModel):
    """Декларативная схема ответа LLM с ограничениями ТЗ из config.py."""

    model_config = ConfigDict(str_strip_whitespace=True)

    seo_title: str = Field(min_length=1, max_length=MAX_TITLE_LEN)
    seo_description: str = Field(min_length=MIN_DESC_LEN, max_length=MAX_DESC_LEN)
    search_queries: list[NonEmptyStr] = Field(min_length=MIN_QUERIES, max_length=MAX_QUERIES)
    commercial_block: CommercialBlockSchema
    notes: list[str] = Field(default_factory=list)


def load_product(path: str | Path) -> dict[str, Any]:
    """Читает JSON-файл товара и валидирует его через ProductSchema."""
    file_path = Path(path)
    if not file_path.is_file():
        raise InputError(f"Файл не найден: {path}")

    try:
        raw_text = file_path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
    except UnicodeDecodeError as err:
        raise InputError(f"Файл {path} должен быть в кодировке UTF-8") from err
    except json.JSONDecodeError as err:
        raise InputError(f"Некорректный JSON в {path} (строка {err.lineno}): {err.msg}") from err

    return validate_product(data)


def validate_product(data: Any) -> dict[str, Any]:
    """Проверяет и нормализует словарь товара с помощью Pydantic."""
    if not isinstance(data, dict):
        raise InputError("Входной JSON должен быть объектом (словарем) с данными товара")
    try:
        return ProductSchema.model_validate(data).model_dump()
    except ValidationError as err:
        fields = ", ".join(".".join(str(x) for x in e["loc"]) for e in err.errors())
        raise InputError(f"Ошибка в полях входного JSON ({fields}): {err.errors()[0]['msg']}") from err


def find_missing(product: dict[str, Any]) -> list[str]:
    """Находит пустые поля по таблице проверяемых ключей (без длинных цепочек if)."""
    top_checks = [
        ("manufacturer", "manufacturer (производитель)"),
        ("description", "description (текстовое описание / паспорт)"),
    ]
    missing = [label for key, label in top_checks if not product.get(key)]

    specs: dict[str, str] = product.get("specs") or {}
    empty_specs = [f"specs.{k}" for k, v in specs.items() if not v]

    if not specs or len(empty_specs) == len(specs):
        return [*missing, "specs (нет заполненных характеристик)"]
    return [*missing, *empty_specs]


def build_key_specs(product: dict[str, Any]) -> list[dict[str, str]]:
    """Детерминированно строит таблицу характеристик с подстановкой «данных нет»."""
    specs: dict[str, str] = product.get("specs") or {}
    spec_items = specs.items() if specs else [("Технические характеристики", "")]

    return [
        {"name": "Производитель", "value": product.get("manufacturer") or NO_DATA},
        *({"name": k, "value": v or NO_DATA} for k, v in spec_items),
    ]


def extract_numbers(text: str) -> set[str]:
    """Извлекает все числа из текста в едином формате (точка как разделитель)."""
    return {match.replace(",", ".") for match in _NUMBER_RE.findall(text)}


def _find_invented_numbers(result: dict[str, Any], product: dict[str, Any]) -> list[str]:
    """Сверяет числа в ответе модели с числами в исходном товаре (без учёта prompt-инъекций)."""
    clean_source = {
        **product,
        "description": _INJECTION_RE.sub(" ", product.get("description") or ""),
    }
    allowed = extract_numbers(json.dumps(clean_source, ensure_ascii=False))
    checked_text = json.dumps(
        [result.get(k) for k in ("seo_title", "seo_description", "search_queries", "commercial_block")],
        ensure_ascii=False,
    )
    return sorted(extract_numbers(checked_text) - allowed)


def _format_pydantic_error(err: dict[str, Any]) -> str:
    """Переводит ошибку Pydantic в понятную подсказку для повторного запроса к LLM."""
    loc = ".".join(str(x) for x in err["loc"])
    return f"поле '{loc}': {err['msg']}"


def check_result(result: dict[str, Any], product: dict[str, Any]) -> tuple[list[str], list[str]]:
    """
    Валидирует ответ LLM декларативно через DraftCardSchema + проверку чисел.
    Возвращает кортеж (ошибки, предупреждения).
    """
    errors: list[str] = []
    try:
        DraftCardSchema.model_validate(result)
    except ValidationError as exc:
        errors.extend(_format_pydantic_error(e) for e in exc.errors())

    invented = _find_invented_numbers(result, product)
    if invented:
        errors.append(f"обнаружены выдуманные числа, которых нет во входе: {', '.join(invented)}")

    return errors, []
