"""Render markdown tables inside the notebook."""
from IPython.display import Markdown, display


def show(md: str):
    """Render a markdown string (one or more tables) in the notebook."""
    display(Markdown(md))