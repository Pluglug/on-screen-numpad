from __future__ import annotations

import os
from typing import TYPE_CHECKING

import bpy

if TYPE_CHECKING:
    from .preferences import OnScreenNumpadPreferences

from .utils.logging import get_logger

ADDON_PATH = os.path.dirname(os.path.abspath(__file__))


def get_prefs(context: bpy.types.Context = None) -> "OnScreenNumpadPreferences":
    """
    Get addon preferences

    Args:
        context: Blender context (optional, defaults to bpy.context)

    Returns:
        OnScreenNumpadPreferences: Addon preferences

    Raises:
        KeyError: If addon preferences cannot be accessed
    """
    if context is None:
        context = bpy.context

    try:
        return context.preferences.addons[__package__].preferences
    except KeyError:
        raise KeyError(f"Addon preferences for '{__package__}' not found")


def init_addon():
    """Initialize addon settings after registration"""
    try:
        prefs = get_prefs()
        log = get_logger()
        if prefs.debug_mode:
            log.set_level("debug")
            log.info("On-Screen Numpad initialized with debug mode enabled")
        else:
            log.set_level("info")
            log.info("On-Screen Numpad initialized")
    except:
        # If preferences not available yet, default to info level
        log = get_logger()
        log.set_level("info")
        log.info("On-Screen Numpad initialized with default settings")
