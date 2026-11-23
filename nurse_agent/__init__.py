"""LangGraph agent for perioperative gynecologic nursing decision support.

    python3 -m nurse_agent "when should the catheter come out after a hysterectomy?"
"""

from .graph import ask, build_agent

__all__ = ["ask", "build_agent"]
