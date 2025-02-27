# LUMI Project Guide

## Commands
- Install dependencies: `uv sync`
- Run: `python -m src.lumi.s2t`
- Add dependency: `uv add package_name`
- Lint: `ruff check .`
- Format: `ruff format .`
- Test: `pytest`
- Test specific file: `pytest path/to/test_file.py`

## Project Setup
- Dependencies require portaudio: `brew install portaudio`
- Speech-to-text is activated by double-tapping Option key
- Audio recordings are saved in temp directory

## Code Standards
- Python >=3.12 required
- Use type hints for functions and classes
- Format imports: standard library first, then third-party, then local
- Exception handling: use try/except with specific exceptions
- Error messages: include context in error messages
- Naming: snake_case for functions/variables, CamelCase for classes
- Documentation: docstrings for functions and classes
- Threading: use proper thread management with cleanup
- Config: use constants at module level for configuration