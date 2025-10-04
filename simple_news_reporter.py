#!/usr/bin/env python3
"""
Simple News Reporter Script

This script takes a single question and generates a brief news report using a web-enabled OpenAI model.
It's a simplified version of the main forecasting system focused on news gathering and reporting.
"""

import argparse
import asyncio
import logging
import os
from datetime import datetime
from openai import AsyncOpenAI
from forecasting_tools import MetaculusApi, MetaculusQuestion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

class SimpleNewsReporter:
    """Simple news reporter that uses OpenAI with web search capabilities."""
    
    def __init__(self, api_key: str = None):
        """Initialize the news reporter with OpenAI API key."""
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
    async def generate_news_report(self, question: str) -> str:
        """
        Generate a news report for the given question using web search.
        
        Args:
            question: The question to research and report on
            
        Returns:
            A formatted news report string
        """
        try:
            logger.info(f"Generating news report for: {question}")
            
            # Create the response with web search enabled
            response = await self.client.responses.create(
                model="gpt-5-mini",  # Web-enabled model
                tools=[{"type": "web_search"}],
                input=f"""You have been tasked with putting together a news report for a forecasting question which will be provided to other forecasters on your team.
                Your job is to find the most recent and relevant news about {question} available on the internet.
                To accomplish this task, you should first think about the question and what types of news or data will be most relevant, then search for those things.
                For questions that involve a time evolving factor (like change in stock price), it may be helpful to report recent history as well as the most current reading.
                Remember, it is not your job to answer the question, but instead to provide a detailed and concise report on key facts.
                Do not offer any suggestions. That is not your job. Instead focus only on reporting the facts.
                Importantly, if there is evidence that the question will resolve imminently, that evidence should be emphasized in the report.
                Format your output in this style (important note, the key facts and high level summary should be the last thing output):
                Question reserached:
                Key considerations to research:
                Key facts found:
                High level summary:""",
            )

            logger.info("News report generated successfully")
            return response.output_text
            
        except Exception as e:
            logger.error(f"Error generating news report: {e}")
            return f"Error generating news report: {str(e)}"
    
    def format_report(self, question: str, report: str, metaculus_question: MetaculusQuestion = None) -> str:
        """
        Format the news report with proper headers and structure.

        Args:
            question: The original question
            report: The generated news report
            metaculus_question: Optional Metaculus question object with additional details

        Returns:
            Formatted news report string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        metaculus_details = ""
        if metaculus_question:
            metaculus_details = f"""
{'='*80}
METACULUS QUESTION DETAILS:
{'='*80}
URL: {metaculus_question.page_url}
Question Type: {type(metaculus_question).__name__}
"""

        formatted_report = f"""
                    {'='*80}
                    SIMPLE NEWS REPORTER
                    Generated: {timestamp}
                    {'='*80}

                    QUESTION:
                    {question}
                    {'='*80}
                    NEWS REPORT:
                    {'='*80}

                    {report}

                    {'='*80}
                    End of Report
                    {'='*80}
                    """
        return formatted_report

async def main():
    """Main function to run the simple news reporter."""
    parser = argparse.ArgumentParser(
        description="Generate a news report for a single question using web search"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="The question to research and report on"
    )
    parser.add_argument(
        "--test-question",
        type=int,
        choices=[1, 2, 3, 4],
        help="Use one of the predefined test questions (1-4)"
    )
    parser.add_argument(
        "--metaculus-id",
        type=int,
        help="Fetch question from Metaculus by question ID"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Save the report to a file (optional)"
    )
    
    args = parser.parse_args()
    
    # Predefined test questions from main.py
    test_questions = {
        1: "Human extinction by 2100",
        2: "Age of oldest human as of 2100", 
        3: "Number of new leading AI labs",
        4: "How many US labor strikes due to AI in 2029"
    }
    
    # Determine which question to use
    metaculus_question = None
    if args.metaculus_id:
        # Fetch question from Metaculus
        logger.info(f"Fetching question {args.metaculus_id} from Metaculus...")
        question_url = f"https://www.metaculus.com/questions/{args.metaculus_id}/"
        try:
            metaculus_question = MetaculusApi.get_question_by_url(question_url)
            question = metaculus_question.question_text
            logger.info(f"Fetched Metaculus question: {question}")
        except Exception as e:
            logger.error(f"Failed to fetch Metaculus question: {e}")
            print(f"Error fetching question from Metaculus: {e}")
            return
    elif args.test_question:
        question = test_questions[args.test_question]
        logger.info(f"Using test question {args.test_question}: {question}")
    elif args.question:
        question = args.question
        logger.info(f"Using custom question: {question}")
    else:
        # Interactive mode - let user choose
        print("\nAvailable test questions:")
        for num, q in test_questions.items():
            print(f"{num}. {q}")
        print("5. Enter custom question")
        
        choice = input("\nSelect a question (1-5): ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= 4:
            question = test_questions[int(choice)]
        elif choice == "5":
            question = input("Enter your custom question: ").strip()
        else:
            print("Invalid choice. Using default question.")
            question = test_questions[3]  # Default to AI labs question
    
    if not question:
        print("No question provided. Exiting.")
        return
    
    # Initialize the reporter
    reporter = SimpleNewsReporter(api_key=args.api_key)
    
    # Generate the news report
    print(f"\nGenerating news report for: {question}")
    print("This may take a moment...")

    report = await reporter.generate_news_report(question)
    formatted_report = reporter.format_report(question, report, metaculus_question)
    
    # Display the report
    print(formatted_report)
    
    # Save to file if requested
    if args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(formatted_report)
            logger.info(f"Report saved to: {args.output_file}")
        except Exception as e:
            logger.error(f"Failed to save report to file: {e}")
    
    # Also save with timestamp if no specific output file
    elif not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"news_report_{timestamp}.txt"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(formatted_report)
            logger.info(f"Report also saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save timestamped report: {e}")

if __name__ == "__main__":
    asyncio.run(main())
