# Newsboy Web Search Integration

The `main.py` file has been enhanced to integrate web search capabilities into the newsboy functionality, similar to the `simple_news_reporter.py` script.

## Key Changes Made

### 1. Added OpenAI Import
```python
from openai import AsyncOpenAI
```

### 2. Enhanced DelphiFall2025 Class
- Added `newsboy_web_search_enabled` configuration flag (default: True)
- Added `_generate_web_search_news()` helper method for web search functionality
- Updated `run_newsboy()` method to use web search when enabled

### 3. New Helper Method
```python
async def _generate_web_search_news(self, question_text: str) -> str:
    """Generate news summary using OpenAI with web search capabilities."""
```

### 4. Enhanced run_newsboy Method
The newsboy now has three modes:
- **Web Search Mode** (default): Uses `gpt-4o-mini-search-preview` with web search tools
- **Regular LLM Mode**: Uses the configured GeneralLlm without web search
- **Disabled Mode**: Returns empty string

### 5. Command Line Control
Added `--disable_newsboy_web_search` flag to control web search behavior.

## Usage Examples

### Enable Web Search (Default)
```bash
python main.py --mode test_questions
```

### Disable Web Search
```bash
python main.py --mode test_questions --disable_newsboy_web_search
```

### With Research Saving
```bash
python main.py --mode test_questions --research_to_txt
```

## How It Works

1. **When web search is enabled** (default):
   - Uses `gpt-4o-mini-search-preview` model with `web_search` tools
   - Provides more recent and accurate news information
   - Focuses on forecasting-relevant developments

2. **When web search is disabled**:
   - Falls back to the configured GeneralLlm
   - Uses the existing prompt-based approach
   - No web search capabilities

3. **Error handling**:
   - If web search fails, it logs the error and returns an error message
   - The research process continues with the error message as news summary

## Benefits

- **More Accurate News**: Web search provides access to recent, real-time information
- **Better Context**: News summaries are more relevant to forecasting questions
- **Configurable**: Can be disabled if needed for testing or rate limiting
- **Backward Compatible**: Existing functionality is preserved
- **Error Resilient**: Graceful fallback on web search failures

## Configuration

The web search feature is controlled by:
- `newsboy_web_search_enabled` class attribute (default: True)
- `--disable_newsboy_web_search` command line flag
- Requires `OPENAI_API_KEY` environment variable

## Integration with Simple News Reporter

The integration uses the same web search approach as `simple_news_reporter.py`:
- Same model: `gpt-4o-mini-search-preview`
- Same tools: `[{"type": "web_search"}]`
- Similar prompt structure focused on forecasting relevance
