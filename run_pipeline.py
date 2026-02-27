"""
Soccer Commentary ASR Data Cleaning Pipeline — Entry Point

Usage:
    python run_pipeline.py                        # process all matches
    python run_pipeline.py --match "Man City"     # single match (partial name match)
    python run_pipeline.py --dry-run              # preview changes, no file writes
    python run_pipeline.py --match "Chelsea" --dry-run  # preview one match

This script coordinates the full Tier 1 + Tier 2 cleaning pipeline:
    1. Discovers all matches with ASR transcription data
    2. For each match: filters hallucinations, removes duplicates,
       detects entities (spaCy NER + heuristics), and corrects
       misspelled names using fuzzy + phonetic matching
    3. Writes cleaned JSON files and a summary report
"""

import argparse
import sys

from pipeline.orchestrator import run_pipeline
from pipeline.report import print_report, save_report


def main():
    parser = argparse.ArgumentParser(
        description="Clean ASR transcriptions of soccer commentary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                          Process all matches
  python run_pipeline.py --match "West Ham"       Process only West Ham match
  python run_pipeline.py --dry-run                Preview without writing files
  python run_pipeline.py --save-report            Save report to file
        """,
    )
    parser.add_argument(
        "--match",
        type=str,
        default=None,
        help="Filter matches by name (partial match, case-insensitive)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing output files",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save the cleaning report to cleaning_report.txt",
    )

    args = parser.parse_args()

    # Run the pipeline
    results = run_pipeline(
        match_filter=args.match,
        dry_run=args.dry_run,
    )

    if not results:
        print("\nNo matches were processed.")
        sys.exit(1)

    # Print the report
    print("\n")
    print_report(results)

    # Optionally save to file
    if args.save_report:
        save_report(results)

    print("\nDone!")


if __name__ == "__main__":
    main()
