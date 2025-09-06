#!/usr/bin/env python3
"""
Script to read forecast report files and extract research and forecast information.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def extract_research_and_forecast(explanation: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract research and forecast sections from the explanation text.
    
    Args:
        explanation: The explanation text containing research and forecast sections
        
    Returns:
        Tuple of (research_text, forecast_text) or (None, None) if not found
    """
    research_text = None
    forecast_text = None
    
    # Extract Research Summary section
    research_match = re.search(r'### Research Summary\n(.*?)(?=\n###|\n##|\Z)', explanation, re.DOTALL)
    if research_match:
        research_text = research_match.group(1).strip()
    
    # Extract Forecasts section
    forecast_match = re.search(r'### Forecasts\n(.*?)(?=\n###|\n##|\Z)', explanation, re.DOTALL)
    if forecast_match:
        forecast_text = forecast_match.group(1).strip()
    
    return research_text, forecast_text


def read_forecast_report(file_path: str) -> List[Dict]:
    """
    Read a single forecast report JSON file and extract research/forecast data.
    
    Args:
        file_path: Path to the JSON report file
        
    Returns:
        List of dictionaries containing question info, research, and forecast data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        
        for item in data:
            if 'question' in item and 'explanation' in item:
                question_text = item['question'].get('question_text', 'Unknown question')
                question_id = item['question'].get('id_of_question', 'Unknown ID')
                
                research, forecast = extract_research_and_forecast(item['explanation'])
                
                result = {
                    'question_id': question_id,
                    'question_text': question_text,
                    'research': research,
                    'forecast': forecast,
                    'file_name': os.path.basename(file_path)
                }
                results.append(result)
        
        return results
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def read_all_forecast_reports(reports_dir: str = "reports") -> List[Dict]:
    """
    Read all forecast report files in the specified directory.
    
    Args:
        reports_dir: Directory containing forecast report files
        
    Returns:
        List of all extracted research and forecast data
    """
    reports_path = Path(reports_dir)
    
    if not reports_path.exists():
        print(f"Reports directory '{reports_dir}' not found.")
        return []
    
    all_results = []
    json_files = list(reports_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in '{reports_dir}' directory.")
        return []
    
    print(f"Found {len(json_files)} JSON files to process:")
    for file_path in json_files:
        print(f"  - {file_path.name}")
    
    for file_path in json_files:
        print(f"\nProcessing {file_path.name}...")
        results = read_forecast_report(str(file_path))
        all_results.extend(results)
        print(f"  Extracted {len(results)} questions from {file_path.name}")
    
    return all_results


def print_results(results: List[Dict], show_research: bool = True, show_forecast: bool = True):
    """
    Print the extracted research and forecast data in a readable format.
    
    Args:
        results: List of extracted data
        show_research: Whether to display research summaries
        show_forecast: Whether to display forecast data
    """
    if not results:
        print("No data to display.")
        return
    
    print(f"\n{'='*80}")
    print(f"FORECAST REPORT ANALYSIS - {len(results)} QUESTIONS FOUND")
    print(f"{'='*80}")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Question ID: {result['question_id']}")
        print(f"   File: {result['file_name']}")
        print(f"   Question: {result['question_text']}")
        
        if show_research and result['research']:
            print(f"\n   RESEARCH SUMMARY:")
            print(f"   {'-'*40}")
            # Truncate very long research summaries for readability
            research_text = result['research']
            # if len(research_text) > 500:
            #     research_text = research_text[:500] + "... [truncated]"
            print(f"   {research_text}")
        
        if show_forecast and result['forecast']:
            print(f"\n   FORECAST:")
            print(f"   {'-'*40}")
            print(f"   {result['forecast']}")
        
        print(f"\n   {'='*60}")


def save_results_to_file(results: List[Dict], output_file: str = "reports/forecast_analysis.txt"):
    """
    Save the extracted data to a text file.
    
    Args:
        results: List of extracted data
        output_file: Output file path
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("FORECAST REPORT ANALYSIS\n")
            f.write("="*50 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"{i}. Question ID: {result['question_id']}\n")
                f.write(f"   File: {result['file_name']}\n")
                f.write(f"   Question: {result['question_text']}\n\n")
                
                if result['research']:
                    f.write("   RESEARCH SUMMARY:\n")
                    f.write("   " + "-"*40 + "\n")
                    f.write(f"   {result['research']}\n\n")
                
                if result['forecast']:
                    f.write("   FORECAST:\n")
                    f.write("   " + "-"*40 + "\n")
                    f.write(f"   {result['forecast']}\n\n")
                
                f.write("   " + "="*60 + "\n\n")
        
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"Error saving results to file: {e}")


def main():
    """Main function to run the forecast report reader."""
    print("Forecast Report Reader")
    print("="*30)
    
    # Read all forecast reports
    results = read_all_forecast_reports()
    
    if not results:
        print("No forecast data found.")
        return
    
    # Print results to console
    print_results(results)
    
    # Ensure reports directory exists
    os.makedirs("reports", exist_ok=True)
    
    # Save results to file
    save_results_to_file(results)
    
    # Print summary statistics
    print(f"\nSUMMARY:")
    print(f"  Total questions processed: {len(results)}")
    research_count = sum(1 for r in results if r['research'])
    forecast_count = sum(1 for r in results if r['forecast'])
    print(f"  Questions with research data: {research_count}")
    print(f"  Questions with forecast data: {forecast_count}")


if __name__ == "__main__":
    main()
