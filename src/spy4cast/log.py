import sys
import os
from typing import Any
from typing import Optional as Op

import datetime

__all__ = [
    "TermColors",
    "log",
    "log_info",
    "log_debug",
    "log_warning",
    "log_error",
]

class TermColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    YELLOW = '\033[33m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log(prefix: str, msg: Any, *args: Any, color: Op[TermColors] = None, info: bool = True, **kwargs: Any) -> None:
    now = datetime.datetime.now()
    if supports_color():
        prefix = f"{TermColors.BOLD}{prefix}{TermColors.ENDC}"
    if info:
        info_str = f" {now:%Y-%m-%d %H:%M:%S}"
    else:
        info_str = f""
    if len(prefix) > 0 or len(info_str) > 0:
        info_str += ":"
    if supports_color() and color is not None:
        prefix = f"{color}{prefix}{color}{info_str}{TermColors.ENDC}"
    s = f"{prefix} {msg}"
    print(s, *args, **kwargs)

def log_info(msg: Any, *args: Any, prefix: str = "INFO", **kwargs: Any) -> None:
    color = kwargs.pop("color", TermColors.HEADER)
    log(prefix, msg, *args, **kwargs, color=color)

def log_debug(msg: Any, *args: Any, prefix: str = "DEBUG", **kwargs: Any) -> None:
    color = kwargs.pop("color", TermColors.YELLOW)
    from . import Settings
    if not Settings.silence:
        log(prefix, msg, *args, **kwargs, color=color)

def log_warning(msg: Any, *args: Any, prefix: str = "WARNING", **kwargs: Any) -> None:
    file = kwargs.pop("file", sys.stderr)
    color = kwargs.pop("color", TermColors.WARNING)
    log(prefix, msg, *args, **kwargs, file=file, color=color)

def log_error(msg: Any, *args: Any, prefix: str = "ERROR", **kwargs: Any) -> None:
    msg2 = f"ERROR: {msg}"
    file = kwargs.pop("file", sys.stderr)
    color = kwargs.pop("color", TermColors.FAIL)
    log(prefix, msg, *args, **kwargs, file=file, color=color)


def supports_color() -> bool:
    """Returns True if the running terminal supports color."""
    # Respect the standard NO_COLOR environment variable
    if os.environ.get("NO_COLOR"):
        return False
        
    # Respect explicit user force overrides
    if os.environ.get("CLICOLOR_FORCE"):
        return True

    # Check if standard output is an interactive terminal (TTY)
    is_a_tty = sys.stdout.isatty()
    
    # On Windows, modern terminals support ANSI, but check platform specifics if needed
    if os.name == "nt":
        # Windows Terminal and modern cmd support colors if not redirected
        return is_a_tty

    # On Unix/Linux/macOS, also check the TERM environment variable
    term = os.environ.get("TERM", "")
    if term in ("dumb", "emacs"):
        return False

    return is_a_tty
