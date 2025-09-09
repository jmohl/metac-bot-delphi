# Simple News Reporter

A simplified script that takes a single question and generates a brief news report using a web-enabled OpenAI model.

## Features

- Takes a single question as input (from predefined test questions or custom input)
- Uses OpenAI's web search capabilities to gather recent news
- Generates a formatted news report
- Saves reports to timestamped files
- Interactive mode for easy question selection

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_simple.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Command Line Options

```bash
python simple_news_reporter.py [options]
```

**Options:**
- `--question "Your custom question"` - Use a custom question
- `--test-question N` - Use predefined test question (1-4)
- `--api-key "your-key"` - Specify API key (or use environment variable)
- `--output-file "filename.txt"` - Save report to specific file

### Examples

1. **Use a predefined test question:**
```bash
python simple_news_reporter.py --test-question 3
```

2. **Use a custom question:**
```bash
python simple_news_reporter.py --question "What are the latest developments in quantum computing?"
```

3. **Interactive mode (no arguments):**
```bash
python simple_news_reporter.py
```

4. **Save to specific file:**
```bash
python simple_news_reporter.py --question "Latest AI developments" --output-file "ai_news.txt"
```

### Predefined Test Questions

1. Human extinction by 2100
2. Age of oldest human as of 2100
3. Number of new leading AI labs
4. How many US labor strikes due to AI in 2029

## Output

The script generates a formatted news report that includes:
- The original question
- Timestamp of generation
- Comprehensive news report with recent developments
- Key facts and context

Reports are automatically saved to timestamped files (e.g., `news_report_20250109_143022.txt`) unless a specific output file is specified.

## Requirements

- Python 3.7+
- OpenAI API key
- Internet connection for web search

## Notes

- The script uses OpenAI's `gpt-4o-mini-search-preview` model with web search capabilities
- Reports are generated asynchronously for better performance
- All output is logged for debugging purposes
