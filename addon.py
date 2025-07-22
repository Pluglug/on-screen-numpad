from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Dict

import bpy

if TYPE_CHECKING:
    from .preferences import OnScreenNumpadPreferences

ADDON_PATH = os.path.dirname(os.path.abspath(__file__))
ADDON_ID = os.path.basename(ADDON_PATH)
ADDON_PREFIX = "".join([s[0] for s in re.split(r"[_-]", ADDON_ID)]).upper()
ADDON_PREFIX_PY = ADDON_PREFIX.lower()

# 検索対象名を正規化
_TARGET_NAME_NORM = ADDON_ID.replace("-", "_")


def _is_target_addon(key: str) -> bool:
    """与えられたアドオンキーがこのアドオンを指すか判定する。

    - `bl_ext.` プレフィックスの有無を無視
    - ハイフン／アンダースコアの差異を無視
    """
    base = key.split(".")[-1]  # 末尾部分
    return base.replace("-", "_") == _TARGET_NAME_NORM


# キャッシュ: key=id(bpy.types.Preferences) -> OnScreenNumpadPreferences
_PREFS_CACHE: Dict[int, "OnScreenNumpadPreferences"] = {}


def get_uprefs(context: bpy.types.Context = bpy.context) -> bpy.types.Preferences:
    """
    Get user preferences

    Args:
        context: Blender context (defaults to bpy.context)

    Returns:
        bpy.types.Preferences: User preferences

    Raises:
        AttributeError: If preferences cannot be accessed
    """
    preferences = getattr(context, "preferences", None)
    if preferences is not None:
        return preferences
    raise AttributeError("Could not access preferences")


def get_prefs(context: bpy.types.Context = bpy.context) -> OnScreenNumpadPreferences:
    """
    Get addon preferences

    Args:
        context: Blender context (defaults to bpy.context)

    Returns:
        OnScreenNumpadPreferences: Addon preferences

    Raises:
        KeyError: If addon is not found
    """
    user_prefs = get_uprefs(context)

    cache_key = id(user_prefs)
    prefs_cached = _PREFS_CACHE.get(cache_key)
    if prefs_cached is not None:
        return prefs_cached

    for key, addon in user_prefs.addons.items():
        if _is_target_addon(key):
            _PREFS_CACHE[cache_key] = addon.preferences
            return addon.preferences

    # 見つからなければエラー
    raise KeyError(f"Addon/Extension '{ADDON_ID}' not found in user preferences.")
