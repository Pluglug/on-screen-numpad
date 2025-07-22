"""
Simple logging utility for quick prototyping.
Usage: log = get_logger(); log.info("message")
"""

from enum import IntEnum
from typing import Optional, TextIO, Union


class LogLevel(IntEnum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


class Log:
    """Simple logger with color output and configurable level/output."""

    _level = LogLevel.DEBUG
    _output: Optional[TextIO] = None
    _colors = {
        LogLevel.DEBUG: "\033[34m",  # blue
        LogLevel.INFO: "\033[1;32m",  # bright green
        LogLevel.WARNING: "\033[33m",  # yellow
        LogLevel.ERROR: "\033[31m",  # red
    }

    @classmethod
    def set_level(cls, level: Union[str, LogLevel]):
        """Log level: 'debug', 'info', 'warning', 'error'"""
        if isinstance(level, str):
            level_map = {
                "debug": LogLevel.DEBUG,
                "info": LogLevel.INFO,
                "warning": LogLevel.WARNING,
                "error": LogLevel.ERROR,
            }
            cls._level = level_map.get(level.lower(), LogLevel.DEBUG)
        else:
            cls._level = level

    @classmethod
    def set_output(cls, output: Optional[TextIO] = None):
        """Set output destination: None for console, file object for file"""
        cls._output = output

    @classmethod
    def _log(cls, level: LogLevel, *args):
        if level < cls._level:
            return

        msg = ", ".join(str(arg) for arg in args)

        try:
            if cls._output:
                cls._output.write(f"{msg}\n")
                cls._output.flush()
            else:
                print(f"{cls._colors[level]}{msg}\033[0m")
        except (OSError, ValueError):
            # Fallback to print if output fails
            print(f"{cls._colors[level]}{msg}\033[0m")

    @classmethod
    def debug(cls, *args):
        cls._log(LogLevel.DEBUG, *args)

    @classmethod
    def info(cls, *args):
        cls._log(LogLevel.INFO, "")
        cls._log(LogLevel.INFO, *args)

    @classmethod
    def warning(cls, *args):
        cls._log(LogLevel.WARNING, *args)

    warn = warning

    @classmethod
    def error(cls, *args):
        cls._log(LogLevel.ERROR, *args)


def get_logger(_name: str = "default"):
    """Standard logger interface for future replacement"""
    return Log
