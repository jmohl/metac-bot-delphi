import argparse
import asyncio
import logging
import os
from datetime import datetime
from tkinter.constants import TRUE
from typing import Literal
from asknews_sdk import AsyncAskNewsSDK
from openai import AsyncOpenAI
import litellm

from asknews_searcher_dp import AskNewsSearcher
from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

logger = logging.getLogger(__name__)

# Register custom model costs for litellm
litellm.register_model({
    "openrouter/openai/gpt-5": {
        "input_cost_per_token": 0.00000125, 
        "output_cost_per_token": 0.00001
    },
    "openrouter/openai/o4-mini": {
        "input_cost_per_token": 0.00000110, 
        "output_cost_per_token": 0.00000440
    },
})


class DelphiFall2025(ForecastBot):
    """
    This is the Delphi Fall 2025 bot.
    This bot is a fork of the Fall 2025 metac template bot with the following changes:
    - Using web enabled openai models for news queries, through my own api
    """

    _max_concurrent_questions = (
        4  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Enable web search for newsboy by default
        self.newsboy_web_search_enabled = getattr(self, 'newsboy_web_search_enabled', True)

    async def _generate_web_search_news(self, question_text: str) -> str:
        """
        Generate news summary using OpenAI with web search capabilities.
        """
        try:
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = await client.responses.create(
                model="gpt-4o-mini",  # Web-enabled model
                tools=[{"type": "web_search"}],
                input=f"""Find the most recent and relevant news about: {question_text}. 
                Provide a concise summary focusing on key developments and facts that would be relevant for forecasting this question.
                In particular, if the question is one that might soon resolve based on observed information, this should be prioritized above all else.
                """,
            )
            
            return response.output_text
        except Exception as e:
            logger.error(f"Web search news generation failed: {e}")
            return f"Web search error: {str(e)}"

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            logger.info(f"Starting research for URL {question.page_url}")
            research = ""
            researcher = self.get_llm("researcher")

            news_summary = await self._generate_web_search_news(question.question_text)

            prompt = clean_indents(
                f"""
                You are an experienced Research Lead on a small superforecasting team.
                You will be given a question to research, for which your team will need to produce a forecast.
                Your role is to provide a concise but detailed rundown of the most relevant background and news, including if the question would resolve Yes or No based on current information.
                If there is news that suggest resolution of the question is imminent then this should be prioritized above all else and strongly emphasized in your report.

                You do not produce forecasts yourself.

                Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}

                The options are: {getattr(question, 'options', 'yes/no or a range of values')}
                If there are multiple options listed, you must provide a brief description of each of the options to provide context.

                You have already tasked one of your junior analysts to produce a recent news summary related to this question.
                They reported the following:
                {news_summary if news_summary else "No initial news summary available."}

                For your research report, first provide a detailed summary of your research as it relates to the question (500-1000 words). 
                Then list the 5-10 most important facts upon which this summary is based.

                """
            )

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif not researcher or researcher == "None":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            #logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            
            # Save research to txt file if research_to_txt is enabled
            if hasattr(self, 'research_to_txt') and self.research_to_txt:
                self._save_research_to_txt(question, prompt, research, news_summary)
            
            return research

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        #logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}"
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            Note that all of the chosen probabilities must be between 0.001 (or 0.1%) and 0.999 (or 99.9%) and that these options MUST SUM to 1.0 EXACTLY.
            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}
            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        #logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}"
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        #logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.nominal_upper_bound is not None:
            upper_bound_number = question.nominal_upper_bound
        else:
            upper_bound_number = question.upper_bound
        if question.nominal_lower_bound is not None:
            lower_bound_number = question.nominal_lower_bound
        else:
            lower_bound_number = question.lower_bound

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number}."
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {upper_bound_number}."
            )

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number}."
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {lower_bound_number}."
            )
        return upper_bound_message, lower_bound_message

    def _save_research_to_txt(self, question: MetaculusQuestion, prompt: str, research: str, news_summary: str) -> None:
        """Save research string to a txt file in the reports folder."""
        try:
            # Create reports directory if it doesn't exist
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)
            
            # Create a unique filename with timestamp and question info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            question_id = question.page_url.split("/")[-2] if "/" in question.page_url else "unknown"
            filename = f"research_{question_id}_{timestamp}.txt"
            filepath = os.path.join(reports_dir, filename)
            
            # Write research content to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Question: {question.question_text}\n")
                f.write(f"URL: {question.page_url}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n" + "=" * 80 + "\n\n")
                f.write("=== RESEARCHER PROMPT ===\n")
                f.write(prompt)
                f.write("\n" + "=" * 80 + "\n\n")
                f.write("=== RESEARCHER OUTPUT ===\n")
                f.write(research)
            
            logger.info(f"Research saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save research to txt file: {e}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the Delphi Fall 2025 forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    parser.add_argument(
        "--research_to_txt",
        action="store_true",
        help="Save research strings to txt files in the reports folder",
    )
    parser.add_argument(
        "--disable_newsboy_web_search",
        action="store_true",
        help="Disable web search for newsboy (use regular LLM instead)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    assert run_mode in [
        "tournament",
        "metaculus_cup",
        "test_questions",
    ], "Invalid run mode"

    delphi_bot = DelphiFall2025(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to="reports",
        skip_previously_forecasted_questions=True,
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-5", # "anthropic/claude-3-5-sonnet-20241022", etc (see docs for litellm)
                temperature=1,
                timeout=180,
                allowed_tries=2,
            ),
            "summarizer": "openrouter/openai/o4-mini",#note, can append openrouter/openai/ to the model name to use OpenRouter. 
            "researcher": "openrouter/openai/gpt-5",
            #"factchecker": "openai/o3-mini",
            "parser": "openrouter/openai/o4-mini",
        },
    )
    
    # Set the research_to_txt flag
    delphi_bot.research_to_txt = args.research_to_txt
    
    # Set the newsboy web search flag
    delphi_bot.newsboy_web_search_enabled = not args.disable_newsboy_web_search

    if run_mode == "tournament":
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
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        # The Metaculus cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564 or AI_2027_TOURNAMENT_ID = "ai-2027"
        # The Metaculus cup may not be initialized near the beginning of a season (i.e. January, May, September)
        delphi_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            delphi_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029 - Discrete
        ]
        delphi_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            delphi_bot.forecast_questions(questions, return_exceptions=True)
        )
    delphi_bot.log_report_summary(forecast_reports)
