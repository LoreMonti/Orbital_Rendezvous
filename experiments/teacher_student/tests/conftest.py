"""Make the study's package, experiments/teacher_student/teacher_student, importable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
