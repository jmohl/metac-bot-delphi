# Tournament ID System - How It Works

## Overview

The Metaculus bot uses tournament IDs to target specific forecasting competitions. The system is designed to automatically track the current season's tournaments without requiring code updates.

## Current Tournament IDs (as of forecasting-tools v0.2.74)

From the highlighted code in `main.py:532-541`:

```python
seasonal_tournament_reports = asyncio.run(
    delphi_bot.forecast_on_tournament(
        MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
    )
)
minibench_reports = asyncio.run(
    delphi_bot.forecast_on_tournament(
        MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True
    )
)
```

These constants resolve to:
- **`CURRENT_AI_COMPETITION_ID`**: `32813` (Fall 2025 AI Benchmarking)
- **`CURRENT_MINIBENCH_ID`**: `"minibench"` (persistent string-based slug)
- **`CURRENT_METACULUS_CUP_ID`**: `32828` (Fall 2025 Metaculus Cup)

## How the System Works

### 1. Constants Defined in forecasting-tools Library

The tournament IDs are maintained as class attributes in the `MetaculusApi` class within the `forecasting-tools` library (`metaculus_api.py`). This includes:

**Historical Tournament IDs:**
- `AI_WARMUP_TOURNAMENT_ID = 3294`
- `AI_COMPETITION_ID_Q3 = 3349`
- `AI_COMPETITION_ID_Q4 = 32506`
- `AI_COMPETITION_ID_Q1 = 32627`
- `AI_COMPETITION_ID_Q2 = 32721`
- `AIB_FALL_2025_ID = 32813`
- `Q3_2024_QUARTERLY_CUP = 3366`
- `Q4_2024_QUARTERLY_CUP = 3672`
- `Q1_2025_QUARTERLY_CUP = 32630`
- `METACULUS_CUP_2025_1_ID = 32726`
- `METACULUS_CUP_FALL_2025_ID = 32828`

**Current Season Aliases:**
```python
CURRENT_AI_COMPETITION_ID = AIB_FALL_2025_ID  # 32813
CURRENT_METACULUS_CUP_ID = METACULUS_CUP_FALL_2025_ID  # 32828
CURRENT_MINIBENCH_ID = "minibench"  # String-based slug
```

### 2. Two Types of Tournament Identifiers

**Integer IDs**: Specific tournament instances with unique numeric IDs
- Example: `32813` for Fall 2025 AI Benchmarking
- These change each season

**String Slugs**: Persistent, season-independent identifiers
- Example: `"minibench"` always points to the current minibench tournament
- Example: `"metaculus-cup"` could be used instead of the integer ID
- Example: `"ai-2027"` for the AI 2027 tournament
- These remain constant across seasons

### 3. Automatic Season Updates

The `CURRENT_*` constants act as **aliases** that the forecasting-tools maintainers update each season. This means:

1. Bots using `MetaculusApi.CURRENT_AI_COMPETITION_ID` automatically target the current season
2. No code changes needed in individual bot repositories when seasons change
3. Simply updating the `forecasting-tools` dependency pulls in new tournament IDs

### 4. How forecast_on_tournament() Uses These IDs

When you call:
```python
delphi_bot.forecast_on_tournament(MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True)
```

The `forecast_on_tournament()` method (inherited from `ForecastBot`):
1. Receives the tournament ID (integer or string)
2. Makes API calls to Metaculus using that ID to fetch all open questions in the tournament
3. Filters questions based on bot configuration (e.g., `skip_previously_forecasted_questions`)
4. Processes each question through the research and forecasting pipeline
5. Publishes forecasts back to Metaculus

## Comparison with main_with_no_framework.py

The standalone implementation defines its own constants locally:

```python
Q4_2024_AI_BENCHMARKING_ID = 32506
Q1_2025_AI_BENCHMARKING_ID = 32627
FALL_2025_AI_BENCHMARKING_ID = "fall-aib-2025"
CURRENT_MINIBENCH_ID = "minibench"
```

This approach requires **manual updates** when seasons change, unlike the forecasting-tools approach.

## Benefits of This System

1. **Automatic Updates**: Upgrade `forecasting-tools` to get new season IDs
2. **Consistency**: All bots using the library get synchronized tournament targeting
3. **Flexibility**: Can use either string slugs (persistent) or integer IDs (specific)
4. **Historical Access**: Old tournament IDs remain available for backtesting

## How to Check Current Values

```bash
poetry run python3 -c "from forecasting_tools import MetaculusApi; \
  print('CURRENT_AI_COMPETITION_ID:', MetaculusApi.CURRENT_AI_COMPETITION_ID); \
  print('CURRENT_MINIBENCH_ID:', MetaculusApi.CURRENT_MINIBENCH_ID); \
  print('CURRENT_METACULUS_CUP_ID:', MetaculusApi.CURRENT_METACULUS_CUP_ID)"
```

## Note on Tournament URLs

Tournament IDs correspond to Metaculus URLs:
- Integer ID `32813` → `https://www.metaculus.com/tournament/fall-aib-2025/`
- String slug `"minibench"` → `https://www.metaculus.com/tournament/minibench/`

The Metaculus API accepts both formats interchangeably (as noted in the source code comment: "The tournament slug can be used for ID as well").

---

## Question Filtering System - How It Pulls Only Open Questions

### Overview

When fetching questions from a tournament, the system uses an `ApiFilter` to ensure only relevant questions are retrieved. For tournament forecasting, the key requirement is fetching only **open** questions (excluding closed, resolved, or upcoming questions).

### The Core Method: `get_all_open_questions_from_tournament()`

Located in `MetaculusApi` class (`metaculus_api.py:366-381`):

```python
@classmethod
def get_all_open_questions_from_tournament(
    cls,
    tournament_id: int | str,
    group_question_mode: GroupQuestionMode = "unpack_subquestions",
) -> list[MetaculusQuestion]:
    logger.info(f"Retrieving questions from tournament {tournament_id}")
    api_filter = ApiFilter(
        allowed_tournaments=[tournament_id],
        allowed_statuses=["open"],
        group_question_mode=group_question_mode,
    )
    questions = asyncio.run(cls.get_questions_matching_filter(api_filter))
    logger.info(
        f"Retrieved {len(questions)} questions from tournament {tournament_id}"
    )
    return questions
```

### How ApiFilter Works

The `ApiFilter` class (lines 49-82) is a Pydantic model that defines filtering criteria for question retrieval:

**Key Filter Parameters:**
```python
class ApiFilter(BaseModel):
    allowed_types: list[QuestionBasicType] = ["binary", "numeric", "multiple_choice", "date", "discrete"]
    allowed_statuses: list[QuestionStateAsString] | None = None  # ["open", "upcoming", "resolved", "closed"]
    allowed_tournaments: list[str | int] | None = None
    group_question_mode: GroupQuestionMode = "exclude"  # or "unpack_subquestions"
    num_forecasters_gte: int | None = None
    community_prediction_exists: bool | None = None
    scheduled_resolve_time_gt: datetime | None = None
    scheduled_resolve_time_lt: datetime | None = None
    order_by: str = "-published_time"
    # ... and many more filtering options
```

### Two-Stage Filtering Process

The filtering happens in **two stages**:

#### 1. API-Level Filtering (Server-Side)

Implemented in `_create_url_params_for_search()` (lines 673-718):

```python
def _create_url_params_for_search(cls, api_filter: ApiFilter, offset: int = 0) -> dict[str, Any]:
    url_params = {
        "limit": 100,  # MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST
        "offset": offset,
        "order_by": api_filter.order_by,
        "with_cp": "true",  # Include community prediction
    }

    if api_filter.allowed_statuses:
        url_params["statuses"] = api_filter.allowed_statuses  # e.g., ["open"]

    if api_filter.allowed_tournaments:
        url_params["tournaments"] = api_filter.allowed_tournaments  # e.g., [32813]

    if api_filter.allowed_types:
        url_params["forecast_type"] = api_filter.allowed_types

    # ... additional time-based filters

    return url_params
```

These parameters are sent to the Metaculus API endpoint: `https://www.metaculus.com/api/posts/`

**For tournament open questions, this translates to:**
```
GET /api/posts/?tournaments=[32813]&statuses=["open"]&limit=100&offset=0&with_cp=true
```

#### 2. Local Filtering (Client-Side)

After receiving results from the API, additional filtering is applied locally in `_apply_local_filters()` (lines 721-770):

```python
def _apply_local_filters(cls, input_questions: list[MetaculusQuestion], api_filter: ApiFilter) -> list[MetaculusQuestion]:
    questions = copy.deepcopy(input_questions)

    if api_filter.allowed_statuses:
        questions = cls._filter_by_status(questions, api_filter.allowed_statuses)

    if api_filter.allowed_types:
        questions = cls._filter_questions_by_type(questions, api_filter.allowed_types)

    if api_filter.num_forecasters_gte is not None:
        questions = cls._filter_questions_by_forecasters(questions, api_filter.num_forecasters_gte)

    if api_filter.community_prediction_exists is not None:
        questions = cls._filter_questions_by_community_prediction_exists(questions, ...)

    # ... additional local filters

    return questions
```

**Status filtering implementation:**
```python
def _filter_by_status(cls, questions: list[Q], statuses: list[QuestionStateAsString]) -> list[Q]:
    return [
        question
        for question in questions
        if question.state is not None and question.state.value in statuses
    ]
```

### Why Two-Stage Filtering?

1. **API-level filtering** reduces data transfer by having the Metaculus server return only relevant questions
2. **Local filtering** handles edge cases where:
   - Some filter criteria aren't supported by the Metaculus API
   - API filtering isn't perfectly accurate (e.g., concurrent state changes)
   - Additional validation is needed on returned data

### Question States

Questions can have the following states:
- **`"open"`**: Currently accepting forecasts (this is what tournaments target)
- **`"upcoming"`**: Published but not yet open for forecasting
- **`"closed"`**: No longer accepting forecasts but not yet resolved
- **`"resolved"`**: Question has been resolved with a final outcome

### Complete Flow for Tournament Question Fetching

```
forecast_on_tournament(tournament_id)
    └─> MetaculusApi.get_all_open_questions_from_tournament(tournament_id)
        └─> Create ApiFilter(allowed_tournaments=[tournament_id], allowed_statuses=["open"])
            └─> get_questions_matching_filter(api_filter)
                └─> _filter_sequential_strategy(api_filter, num_questions=None)
                    └─> Loop through pages (if needed):
                        └─> _grab_filtered_questions_with_offset(api_filter, offset)
                            └─> _create_url_params_for_search(api_filter, offset)
                                ├─> Creates: {tournaments: [32813], statuses: ["open"], limit: 100, offset: 0}
                            └─> _get_questions_from_api(url_params, group_question_mode)
                                ├─> Sleep 2-3 seconds (rate limiting)
                                ├─> GET /api/posts/ with url_params
                                └─> Parse JSON response into MetaculusQuestion objects
                            └─> _apply_local_filters(questions, api_filter)
                                └─> Verify status is "open" locally
                                └─> Apply any additional local filters
```

### Group Question Handling

The `group_question_mode` parameter controls how group questions (questions with multiple subquestions) are handled:

- **`"exclude"`**: Group questions are completely removed from results
- **`"unpack_subquestions"`**: Each subquestion is extracted and treated as a separate question

For tournaments, the default is **`"unpack_subquestions"`** so the bot can forecast on each subquestion individually.

### Pagination and Rate Limiting

- Questions are fetched in **batches of 100** (the API maximum)
- Each batch request includes a **2-3 second delay** (see rate limiting documentation above)
- For tournaments with >100 open questions, multiple paginated requests are made
- The `offset` parameter advances by 100 for each subsequent request

### Example: Fetching Fall 2025 AI Benchmarking Questions

```python
# Inside forecast_on_tournament()
questions = MetaculusApi.get_all_open_questions_from_tournament(32813)

# This creates the filter:
api_filter = ApiFilter(
    allowed_tournaments=[32813],
    allowed_statuses=["open"],
    group_question_mode="unpack_subquestions"
)

# Which generates API request:
# GET /api/posts/?tournaments=[32813]&statuses=["open"]&limit=100&offset=0&with_cp=true

# Returns only open questions from tournament 32813
```

### Benefits of This Filtering Approach

1. **Efficiency**: Server-side filtering reduces network bandwidth
2. **Reliability**: Local filtering catches edge cases and validates data
3. **Flexibility**: Easy to add new filter criteria without API changes
4. **Correctness**: Two-stage approach ensures only intended questions are processed
5. **Maintainability**: Filter logic is centralized in the ApiFilter class

### Customizing Filters

While tournament forecasting uses `allowed_statuses=["open"]`, you can customize filters for other use cases:

```python
# Get all questions (including closed) from a tournament
api_filter = ApiFilter(
    allowed_tournaments=[32813],
    allowed_statuses=["open", "closed", "resolved"]
)

# Get only binary questions with high engagement
api_filter = ApiFilter(
    allowed_tournaments=[32813],
    allowed_statuses=["open"],
    allowed_types=["binary"],
    num_forecasters_gte=50
)

# Get questions resolving within 30 days
api_filter = ApiFilter(
    allowed_tournaments=[32813],
    allowed_statuses=["open"],
    scheduled_resolve_time_lt=datetime.now() + timedelta(days=30)
)
```
