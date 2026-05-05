from pathlib import Path
import sys


def setup_project_imports():
    """
    Wykrywa główny folder projektu i dodaje go do ścieżki systemowej Pythona,
    """
    project_root = Path(__file__).resolve().parent.parent
    
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)