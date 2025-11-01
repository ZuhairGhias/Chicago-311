# Chicago 311 Response Time Analysis

> **Scaffolding note:** The initial project structure was generated collaboratively with OpenAI's ChatGPT (Codex).

This repository provides a lightweight foundation for exploring the City of Chicago 311 service request data and building
response-time models. The goal is to keep the structure simple while leaving room to grow as the team adds code and
notebooks.

## Repository Layout

```
├── data/                 # Local data storage (ignored by git)
├── notebooks/            # Prototyping notebooks
├── src/                  # Reusable project code
│   ├── config.py
│   ├── methods/
│   ├── processing/
│   └── visualizations/
├── tests/                # Basic smoke tests
└── pyproject.toml        # Minimal dependency definition
```

## Quick Start

1. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

2. **Open the starter notebook**
   ```bash
   jupyter notebook notebooks/chicago_311_analysis.ipynb
   ```

3. **Add your own modules and analyses**
   - Keep reusable functions inside `src/`.
   - Use notebooks for experiments, committing only the versions needed for collaboration.

## Next Steps

The configuration files are intentionally minimal. As the project evolves, add formatting, linting, or additional
dependencies as needed for the team.
