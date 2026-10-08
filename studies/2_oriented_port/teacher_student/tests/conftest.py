"""Make the study's package, studies/2_oriented_port/teacher_student/teacher_student, importable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
