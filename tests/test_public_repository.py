import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT_FILES = {
    "README.md",
    "REPRODUCIBILITY.md",
    "CONTRIBUTING.md",
    "CITATION.cff",
    "results/README.md",
    "pyproject.toml",
    "requirements-lock.txt",
}
PUBLIC_DIRECTORIES = ("src", "configs", "data", "tests", ".github")
TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
EXCLUDED_PARTS = {"raw", "processed", "__pycache__", ".pytest_cache"}
CHINESE_CHARACTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
USER_ABSOLUTE_PATH = re.compile(r"/(?:home|Users)/[^\s/'\"]+")


def public_text_files() -> list[Path]:
    files = [REPOSITORY_ROOT / name for name in PUBLIC_ROOT_FILES]
    for directory_name in PUBLIC_DIRECTORIES:
        directory = REPOSITORY_ROOT / directory_name
        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in TEXT_SUFFIXES
            and not EXCLUDED_PARTS.intersection(path.relative_to(directory).parts)
        )
    return sorted(set(files))


def test_public_text_is_english_and_portable() -> None:
    violations: list[str] = []
    for path in public_text_files():
        text = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(REPOSITORY_ROOT)
        for line_number, line in enumerate(text.splitlines(), start=1):
            if CHINESE_CHARACTER.search(line):
                violations.append(f"{relative_path}:{line_number}: Chinese-language text")
            if WINDOWS_ABSOLUTE_PATH.search(line) or USER_ABSOLUTE_PATH.search(line):
                violations.append(f"{relative_path}:{line_number}: machine-specific absolute path")

    assert not violations, "Public repository policy violations:\n" + "\n".join(violations)
