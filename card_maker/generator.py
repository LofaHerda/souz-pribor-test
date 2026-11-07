"""Генерация карточки товара через OpenAI SDK, самокоррекция и рендеринг в Markdown."""
from __future__ import annotations

import json
from typing import Any, Callable

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from .config import (
    LLM_TEMPERATURE,
    MAX_ATTEMPTS,
    MAX_DESC_LEN,
    MAX_TITLE_LEN,
    MIN_DESC_LEN,
    NO_DATA,
    TARGET_DESC_LEN,
    AppConfig,
    ConfigError,
    LLMClientFactory,
)
from .prompts import SYSTEM_PROMPT, VERIFY_PROMPT, build_user_prompt
from .validation import build_key_specs, check_result, find_missing


class GenerationError(Exception):
    """Ошибка конфигурации клиента или взаимодействия с LLM API."""

_API_ERROR_HANDLERS: dict[type[Exception], Callable[[Any], str]] = {
    AuthenticationError: lambda _: "Ошибка авторизации API (401/403): проверьте API-ключ в .env",
    RateLimitError: lambda _: "Превышен лимит запросов к API (429) или исчерпана квота",
    APITimeoutError: lambda _: "Превышено время ожидания ответа от LLM API (таймаут)",
    APIConnectionError: lambda e: f"Ошибка сетевого соединения с LLM API: {e}",
    APIStatusError: lambda e: f"Ошибка сервера API (статус {e.status_code}): {e.message}",
}


def make_client(config: AppConfig | None = None) -> tuple[OpenAI, str]:
    try:
        return LLMClientFactory.create(config)
    except ConfigError as err:
        raise GenerationError(str(err)) from err


def parse_json_response(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    candidate = cleaned[start : end + 1] if (0 <= start < end) else cleaned
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Ответ модели должен быть JSON-объектом")
    return parsed


def _call_llm(client: OpenAI, model: str, messages: list[dict[str, str]]) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=LLM_TEMPERATURE,
        )
    except Exception as exc:
        for err_cls, formatter in _API_ERROR_HANDLERS.items():
            if isinstance(exc, err_cls):
                raise GenerationError(formatter(exc)) from exc
        raise GenerationError(f"Непредвиденная ошибка LLM API: {exc}") from exc

    choices = getattr(response, "choices", None)
    content = (choices[0].message.content or "").strip() if choices else ""
    if not content:
        raise GenerationError("Модель вернула пустой ответ")
    return content


def verify_facts_with_llm(
    product: dict[str, Any],
    draft: dict[str, Any],
    client: OpenAI,
    model: str,
) -> list[str]:
    subset = {k: draft.get(k) for k in ("seo_title", "seo_description", "commercial_block")}
    messages = [
        {"role": "system", "content": VERIFY_PROMPT},
        {
            "role": "user",
            "content": (
                f"ИСХОДНЫЕ ДАННЫЕ:\n{json.dumps(product, ensure_ascii=False)}\n\n"
                f"ТЕКСТ КАРТОЧКИ:\n{json.dumps(subset, ensure_ascii=False)}"
            ),
        },
    ]
    try:
        parsed = parse_json_response(_call_llm(client, model, messages))
        items = parsed.get("unsupported")
        return [str(x).strip() for x in items if str(x).strip()][:5] if isinstance(items, list) else []
    except (GenerationError, ValueError):
        return []


def _length_adjustment_hint(desc: Any) -> str | None:
    if not isinstance(desc, str):
        return None
    length = len(desc.strip())
    if length < MIN_DESC_LEN:
        return (
            f"добавь ещё около {TARGET_DESC_LEN - length} симв. в seo_description "
            f"(сейчас {length}, нужно {MIN_DESC_LEN}–{MAX_DESC_LEN})"
        )
    if length > MAX_DESC_LEN:
        return (
            f"сократи seo_description примерно на {length - TARGET_DESC_LEN} симв. "
            f"(сейчас {length}, нужно {MIN_DESC_LEN}–{MAX_DESC_LEN})"
        )
    return None


def _evaluate_reply(
    raw_reply: str,
    product: dict[str, Any],
    client: OpenAI,
    model: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        draft = parse_json_response(raw_reply)
    except ValueError:
        return None, ["ответ не является валидным JSON-объектом"]

    errors, _ = check_result(draft, product)
    if not errors:
        unsupported = verify_facts_with_llm(product, draft, client, model)
        if unsupported:
            errors.append("удали неподтверждённые факты: " + "; ".join(unsupported))

    return draft, errors


def _clean_list(items: Any, default: list[str] | None = None) -> list[str]:
    cleaned = [str(x).strip() for x in (items if isinstance(items, list) else []) if str(x).strip()]
    return cleaned or list(default or [])


def generate_card(product: dict[str, Any], client: OpenAI, model: str) -> dict[str, Any]:
    missing = find_missing(product)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(product, missing)},
    ]

    draft: dict[str, Any] | None = None
    errors: list[str] = []

    for _ in range(MAX_ATTEMPTS):
        raw_reply = _call_llm(client, model, messages)
        draft, errors = _evaluate_reply(raw_reply, product, client, model)
        if not errors:
            break

        len_hint = _length_adjustment_hint(draft.get("seo_description") if draft else None)
        all_hints = [*errors, *([len_hint] if len_hint else [])]
        messages.extend(
            [
                {"role": "assistant", "content": raw_reply},
                {"role": "user", "content": "Исправь замечания и верни JSON целиком: " + "; ".join(all_hints)},
            ]
        )

    if draft is None:
        raise GenerationError("Не удалось получить корректный JSON от модели: " + "; ".join(errors))

    cb = draft.get("commercial_block") if isinstance(draft.get("commercial_block"), dict) else {}
    return {
        "product_name": product["name"],
        "missing_fields": missing,
        "seo_title": str(draft.get("seo_title") or "").strip(),
        "seo_description": str(draft.get("seo_description") or "").strip(),
        "key_specs": build_key_specs(product),
        "search_queries": _clean_list(draft.get("search_queries")),
        "commercial_block": {
            "suitable_for": _clean_list(cb.get("suitable_for"), default=[NO_DATA]),
            "tasks": _clean_list(cb.get("tasks"), default=[NO_DATA]),
        },
        "notes": _clean_list(draft.get("notes")),
        "validation_warnings": errors,
    }


def _bullets(items: list[str], fmt: str = "- {}") -> str:
    return "\n".join(fmt.format(x) for x in (items or [NO_DATA]))


def to_markdown(card: dict[str, Any]) -> str:
    title = card.get("seo_title") or NO_DATA
    desc = card.get("seo_description") or NO_DATA
    cb = card.get("commercial_block") or {}

    specs_rows = "\n".join(
        f"| {str(s['name']).replace('|', '\\|')} | {str(s['value']).replace('|', '\\|')} |"
        for s in (card.get("key_specs") or [])
    )

    banners = [
        (
            f"> **Поля с отсутствующими данными (требуют уточнения):** "
            f"{', '.join(card['missing_fields'])}"
        )
        if card.get("missing_fields")
        else "",
        (
            f"> **Обратите внимание при проверке:** "
            f"{'; '.join(card['validation_warnings'])}"
        )
        if card.get("validation_warnings")
        else "",
    ]

    sections = [
        f"# {card['product_name']}",
        *[b for b in banners if b],
        f"## SEO-заголовок\n**{title}**  \n*Длина: {len(title)} симв. (лимит: до {MAX_TITLE_LEN} симв.)*",
        f"## SEO-описание\n{desc}\n\n*Длина: {len(desc)} симв. (требование: {MIN_DESC_LEN}–{MAX_DESC_LEN} симв.)*",
        f"## Ключевые характеристики\n| Характеристика | Значение |\n|---|---|\n{specs_rows}",
        f"## Поисковые запросы (SEO)\n{_bullets(card.get('search_queries') or [], fmt='- `{}`')}",
        (
            f"## Commercial-блок для менеджера\n"
            f"### Кому подходит прибор\n{_bullets(cb.get('suitable_for') or [NO_DATA])}\n\n"
            f"### Какие задачи решает\n{_bullets(cb.get('tasks') or [NO_DATA])}"
        ),
    ]

    if card.get("notes"):
        sections.append(f"## Замечания и вопросы к поставщику\n{_bullets(card['notes'])}")

    return "\n\n".join(sections) + "\n"
