# Contributing to Mosaic

Thank you for your interest in contributing to Mosaic! We welcome contributions of all kinds.

## Getting Started

1. Fork the repository and clone it locally.
2. Create a virtual environment:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   ```
3. Install in editable mode with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

- Create a feature branch from `main`: `git checkout -b feat/my-feature`
- Make your changes and add tests.
- Run linting: `ruff check mosaic/ tests/`
- Run type checking: `mypy mosaic/`
- Run tests: `pytest`
- Commit with a descriptive message.
- Open a pull request against `main`.

## Code Style

- Follow PEP 8, enforced by `ruff`.
- Use type hints for all public APIs.
- Write docstrings in Google style.
- Keep line length to 100 characters.

## Testing

- Write tests for all new features and bug fixes.
- Place tests in `tests/`, mirroring the `mosaic/` source structure.
- Run `pytest` before submitting a PR.

## Pull Request Process

1. Ensure all CI checks pass.
2. Update documentation if applicable.
3. Add an entry to `CHANGELOG.md` under the `[Unreleased]` section.
4. Request review from a maintainer.

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.