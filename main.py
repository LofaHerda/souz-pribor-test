"""Точка входа CLI: python main.py input_product.json"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

from card_maker.config import NO_DATA, PROJECT_ROOT, AppConfig, ConfigError, LLMClientFactory, load_env_file
from card_maker.generator import GenerationError, generate_card, to_markdown
from card_maker.validation import InputError, build_key_specs, find_missing, load_product

DEFAULT_INPUT = PROJECT_ROOT / "input_product.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="CLI-генератор черновиков карточек товаров КИПиА в формате Markdown.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Путь к JSON-файлу (например, input_product.json) или папке с JSON-файлами",
    )
    parser.add_argument(
        "-o",
        "--out",
        default="output",
        help="Папка для сохранения готовых Markdown-карточек (по умолчанию: output)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Только проверить входные JSON-файлы без обращения к LLM API",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Не печатать карточку в консоль (только сохранить в файл)",
    )
    return parser


def resolve_input_paths(raw_paths: list[str]) -> list[str]:
    """Разворачивает список путей; если аргументы не переданы, берёт input_product.json."""
    if not raw_paths:
        return [str(DEFAULT_INPUT)] if DEFAULT_INPUT.is_file() else []

    resolved: list[str] = []
    for item in raw_paths:
        path = Path(item)
        resolved.extend(str(f) for f in sorted(path.glob("*.json"))) if path.is_dir() else resolved.append(item)
    return resolved


def print_check_report(path: str, product: dict[str, Any], missing: list[str]) -> None:
    """Печатает сводку проверки входного JSON-файла (режим --check)."""
    status = (
        "⚠ Требуют уточнения: " + ", ".join(missing)
        if missing
        else "✓ Все основные поля и характеристики заполнены."
    )
    specs_preview = "\n".join(f"  • {r['name']}: {r['value']}" for r in build_key_specs(product))
    print(
        f"\n=== Проверка файла: {path} ===\n"
        f"Товар: {product['name']}\n"
        f"Производитель: {product['manufacturer'] or NO_DATA}\n"
        f"{status}\n"
        f"Таблица характеристик:\n{specs_preview}"
    )


def _process_single_file(
    path: str,
    args: argparse.Namespace,
    client_pair: tuple[OpenAI, str] | None,
) -> tuple[OpenAI, str] | None:
    """Обрабатывает один JSON-файл товара (проверка или генерация + сохранение)."""
    product = load_product(path)
    missing = find_missing(product)

    if args.check:
        print_check_report(path, product, missing)
        return client_pair

    if missing:
        print(f"[{path}] ⚠ Пустые поля (укажем «данных нет»): {', '.join(missing)}", file=sys.stderr)

    client, model = client_pair or LLMClientFactory.create()
    print(f"[{path}] Генерация карточки через LLM ({model})...", flush=True)

    md_content = to_markdown(generate_card(product, client, model))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{Path(path).stem}.md"
    out_file.write_text(md_content, encoding="utf-8")

    if not args.quiet:
        print("\n" + md_content)
    print(f"[{path}] ✓ Сохранено в {out_file}")
    return client, model


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    load_env_file()

    files = resolve_input_paths(args.paths)
    if not files:
        print("Укажите путь к JSON-файлу: python main.py input_product.json", file=sys.stderr)
        return 1

    failed = 0
    client_pair: tuple[OpenAI, str] | None = None

    for path in files:
        try:
            client_pair = _process_single_file(path, args, client_pair)
        except (InputError, GenerationError, ConfigError) as err:
            print(f"[{path}] ОШИБКА: {err}", file=sys.stderr)
            failed += 1
            if isinstance(err, (GenerationError, ConfigError)) and client_pair is None:
                break

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
