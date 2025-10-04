# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a forecasting bot for Metaculus built on the `forecasting-tools` framework. The bot (`DelphiFall2025`) makes predictions on binary, multiple choice, numeric, and discrete questions by conducting research and generating reasoned forecasts.

## Core Architecture

### Main Bot Class: `DelphiFall2025` (main.py)

The bot inherits from `ForecastBot` and implements three main phases:

1. **Research Phase** (`run_research`)
   - Uses OpenAI's web-enabled models (gpt-4o-mini) to generate news summaries via `_generate_web_search_news`
   - Passes news summary and question details to a researcher LLM
   - Outputs 500-1000 word research reports with 5-10 key facts
   - Can optionally save research to txt files in `reports/` directory

2. **Forecasting Phase** (question-type specific methods)
   - `_run_forecast_on_binary`: Returns probability (0.01-0.99)
   - `_run_forecast_on_multiple_choice`: Returns probability distribution over options
   - `_run_forecast_on_numeric`: Returns percentile distribution (10th, 20th, 40th, 60th, 80th, 90th)
   - Each uses status quo bias and tournament-aware adjustments

3. **LLM Configuration**
   - Uses litellm with custom model cost registration for OpenRouter models
   - Supports multiple LLM roles: default, summarizer, researcher, parser
   - Concurrency controlled via `_concurrency_limiter` (default: 4 concurrent questions)

### Supporting Files

- `asknews_searcher_dp.py`: AskNewsSearcher class for news retrieval (currently not actively used in main bot)
- `community_benchmark.py`: Benchmarking system comparing bot forecasts to community predictions
- `read_forecast_reports.py`: Utilities for reading saved forecast reports
- `simple_news_reporter.py`: Alternative news reporting implementation
- `main_with_no_framework.py`: Standalone implementation without forecasting-tools framework

## Development Commands

### Running the Bot

```bash
# Test on example questions (recommended for development)
poetry run python main.py --mode test_questions

# Run on current tournament
poetry run python main.py --mode tournament

# Run on Metaculus Cup
poetry run python main.py --mode metaculus_cup

# Save research reports to txt files
poetry run python main.py --mode test_questions --research_to_txt

# Disable web search for news (use regular LLM)
poetry run python main.py --mode test_questions --disable_newsboy_web_search
```

### Benchmarking

```bash
# Run benchmark against community predictions
poetry run python community_benchmark.py --mode run

# Run custom benchmark (e.g., test retrieval by removing background info)
poetry run python community_benchmark.py --mode custom

# View benchmark results in Streamlit UI
poetry run streamlit run community_benchmark.py
```

### Dependencies

```bash
# Install all dependencies
poetry install

# Add new dependency
poetry add <package-name>
```

## Environment Variables

Required secrets (set in `.env` or GitHub Actions secrets):

- `METACULUS_TOKEN`: Authentication for Metaculus API
- `OPENAI_API_KEY`: For web-enabled news generation (gpt-4o-mini)
- `OPENROUTER_API_KEY`: For accessing OpenRouter models (gpt-5, o3, o4-mini)
- `ANTHROPIC_API_KEY`: If using Claude models
- `ASKNEWS_CLIENT_ID` & `ASKNEWS_SECRET`: For AskNews API (if using AskNewsSearcher)
- `PERPLEXITY_API_KEY` & `EXA_API_KEY`: Optional search providers

## GitHub Actions Workflows

Three automated workflows in `.github/workflows/`:

1. `run_bot_on_tournament.yaml`: Runs daily at midnight on seasonal tournament and minibench
2. `run_bot_on_metaculus_cup.yaml`: Runs every 2 days on Metaculus Cup
3. `test_bot.yaml`: Runs on test questions (schedule/trigger varies)

All use concurrency groups to prevent parallel runs.

## Key Implementation Details

### Cost Tracking

- Uses litellm's cost tracking with custom model pricing registered at startup
- OpenRouter models registered: `openrouter/openai/gpt-5` and `openrouter/openai/o4-mini`

### Logging

- Standard logging configured at INFO level
- LiteLLM logger suppressed to WARNING to reduce noise
- Benchmark runs save logs to `benchmarks/log_<timestamp>.log`

### Question Skipping

- By default, `skip_previously_forecasted_questions=True` for tournament mode
- Set to `False` for test_questions and metaculus_cup modes to allow re-forecasting

### Output Files

- Research reports: `reports/` directory
- Benchmark results: `benchmarks/` directory
- Bot forecasts published to Metaculus when `publish_reports_to_metaculus=True`

## Customizing the Bot

To modify forecasting behavior:

1. **Change LLM models**: Edit the `llms` dict in `main.py` __main__ block
2. **Adjust prompts**: Modify prompt strings in `run_research` and `_run_forecast_on_*` methods
3. **Add research tools**: Extend `run_research` to incorporate additional search/analysis tools
4. **Tune predictions**: Adjust `predictions_per_research_report` (default: 5 for tournament, 1 for benchmarking)

## Benchmarking System

The benchmarker (from forecasting-tools) scores bots using expected baseline score against community predictions. Configure multiple bot variants in `community_benchmark.py` to compare:

- Different LLM models
- Different temperature settings
- Different research approaches
- Different numbers of predictions per question

Results include statistical error bars and bot reasoning for each question.
