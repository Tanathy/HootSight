"""Simple script that checks that the UI no longer contains the Dataset tab or page code.

This is a quick, low-risk smoke test you can run after editing UI to ensure the Dataset nav
item, page entries, and the dataset refresh function are gone.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent

def check_file_forbidden_patterns(file_path: Path, patterns: list[str]) -> list[str]:
    try:
        text = file_path.read_text(encoding='utf-8')
    except Exception as e:
        return [f"ERROR: Could not read {file_path}: {e}"]

    found = []
    for p in patterns:
        if p in text:
            found.append(p)
    return found


def main():
    ui_dir = ROOT / 'ui'
    index_html = ui_dir / 'index.html'
    js_dir = ui_dir / 'js'

    errors = []

    # Patterns to check: nav item, dataset-cards and refreshDatasetInfo usage
    patterns = [
        'data-page="dataset"',
        'id="dataset-cards"',
        'refreshDatasetInfo(',
        'buildDatasetPage('
    ]

    files_to_scan = [index_html]
    files_to_scan.extend(sorted(js_dir.glob('**/*.js')))

    for f in files_to_scan:
        issues = check_file_forbidden_patterns(f, patterns)
        if issues:
            errors.append(f"{f}: {', '.join(issues)}")

    if errors:
        print("Dataset UI cleanup FAILED. Found forbidden patterns:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("Dataset UI cleanup PASS — no forbidden patterns found.")

if __name__ == '__main__':
    main()
