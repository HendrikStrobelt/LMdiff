from pathlib import Path
import os

# SERVING STATIC FILES
ROOT = Path(
    os.path.abspath(__file__)
).parent.parent  # Root directory of the project
SRC = ROOT / "src"
CLIENT = ROOT / "client"
DIST = CLIENT / "dist"
TESTS = ROOT / "tests"
DATA = ROOT / "data"
COMPARISONS = DATA / "compared_results"
ANALYSIS = DATA / "analysis_results"
ANALYSIS_DELIM = "_._"
DATASETS = DATA / "datasets"