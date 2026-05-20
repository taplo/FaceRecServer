# Patterns: FaceRecServer

## Current Patterns
- **Minimal scaffold**: Single `main.py` with `if __name__ == "__main__"` guard
- **uv-managed**: Using pyproject.toml (not setup.py or requirements.txt)
- **Type hinting**: Not yet used (code is too minimal)
- **No async**: Not yet used

## Recommendations
- Add type hints from the start
- Use async for the HTTP server
- Keep modules small and focused
