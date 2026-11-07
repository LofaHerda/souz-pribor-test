"""Модульные тесты пакета card_maker и скрипта main.py без реальных вызовов API."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from card_maker.generator import GenerationError, generate_card, make_client, to_markdown
from card_maker.validation import (
    NO_DATA,
    InputError,
    build_key_specs,
    check_result,
    find_missing,
    load_product,
    validate_product,
)
from main import main, resolve_input_paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"


def make_mock_client(payload: dict):
    """Создаёт мок-клиент OpenAI, возвращающий заданный словарь в виде JSON."""

    def create(**kwargs):
        msgs = kwargs.get("messages", [])
        sys_msg = msgs[0]["content"] if msgs else ""
        reply = {"unsupported": []} if "аудитор" in sys_msg else payload
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(reply, ensure_ascii=False)))]
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


VALID_LLM_RESPONSE = {
    "seo_title": "Манометр показывающий DM 02 Манотомь (0–1,6 МПа)",
    "seo_description": (
        "Манометр показывающий DM 02 производства Манотомь предназначен для измерения избыточного "
        "давления неагрессивных жидких и газообразных сред с температурой до 150 °C. "
    )
    * 6,
    "search_queries": [
        "манометр DM 02 купить",
        "манометр 0-1,6 МПа класс 1,5",
        "манометр G1/2 100 мм Манотомь",
    ],
    "commercial_block": {
        "suitable_for": ["Службы эксплуатации котельных и систем водоснабжения"],
        "tasks": ["Контроль избыточного давления в диапазоне 0–1,6 МПа"],
    },
    "notes": [],
}


def test_load_input_product_and_three_examples():
    """Проверяет чтение input_product.json и всех 3 примеров разной сложности."""
    p_root = load_product(PROJECT_ROOT / "input_product.json")
    assert p_root["name"] == "Манометр показывающий DM 02"

    files = sorted(EXAMPLES_DIR.glob("*.json"))
    assert len(files) == 3

    p2 = load_product(files[1])
    missing_p2 = find_missing(p2)
    assert "manufacturer (производитель)" in missing_p2
    assert "specs.класс точности" in missing_p2

    p3 = load_product(files[2])
    assert find_missing(p3) == ["specs.питание"]


def test_pydantic_validation_errors(tmp_path):
    """Проверяет перехват битого JSON и ошибок схемы ProductSchema."""
    bad_file = tmp_path / "broken.json"
    bad_file.write_text("{invalid json", encoding="utf-8")
    with pytest.raises(InputError, match="Некорректный JSON"):
        load_product(bad_file)

    with pytest.raises(InputError, match="name"):
        validate_product({"manufacturer": "Тест", "specs": {}})

    with pytest.raises(InputError, match="specs"):
        validate_product({"name": "Прибор", "specs": ["не словарь"]})


def test_key_specs_deterministic_no_data():
    """Проверяет, что пустые характеристики получают пометку «данных нет»."""
    product = validate_product(
        {
            "name": "Датчик ДРД-М",
            "manufacturer": "",
            "specs": {"питание": "24 В", "класс точности": "", "корпус": None},
            "description": "Описание",
        }
    )
    specs = build_key_specs(product)
    assert {"name": "Производитель", "value": NO_DATA} in specs
    assert {"name": "питание", "value": "24 В"} in specs
    assert {"name": "класс точности", "value": NO_DATA} in specs
    assert {"name": "корпус", "value": NO_DATA} in specs


def test_hallucinated_numbers_and_prompt_injection_blocked():
    """Проверяет, что выдуманные числа и числа из prompt injection в 3_hard.json отлавливаются."""
    p3 = load_product(EXAMPLES_DIR / "3_hard.json")
    injected_resp = dict(
        VALID_LLM_RESPONSE,
        seo_title="Расходомер ПРВ-100-Ex с гарантией 50 лет и точностью 0,01%",
    )
    errors, _ = check_result(injected_resp, p3)
    assert any("50" in e or "0.01" in e for e in errors)


def test_generate_card_and_markdown_output():
    """Проверяет полный цикл сборки карточки и форматирования в Markdown."""
    p1 = load_product(PROJECT_ROOT / "input_product.json")
    client = make_mock_client(VALID_LLM_RESPONSE)
    card = generate_card(p1, client, "mock-model")
    assert card["validation_warnings"] == []
    md = to_markdown(card)
    assert "# Манометр показывающий DM 02" in md
    assert "## Ключевые характеристики" in md


def test_main_cli_check_mode():
    """Проверяет запуск `python main.py input_product.json --check` и `examples/ --check`."""
    assert main([str(PROJECT_ROOT / "input_product.json"), "--check"]) == 0
    assert len(resolve_input_paths([str(EXAMPLES_DIR)])) == 3
    assert main([str(EXAMPLES_DIR), "--check"]) == 0


def test_make_client_env_handling(monkeypatch):
    """Проверяет создание клиента через фабрику LLMClientFactory и AppConfig."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(GenerationError, match="OPENAI_API_KEY"):
        make_client()

    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.endpoint.local/v1")
    monkeypatch.setenv("OPENAI_MODEL", "custom-model")
    client, model = make_client()
    assert model == "custom-model"
    assert "custom.endpoint.local" in str(client.base_url)
