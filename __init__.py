import bpy

from .core import cleanup_on_reload
from .keymaps import register_keymaps, unregister_keymaps
from .menu_integration import cleanup_menu_on_reload, register_menu, unregister_menu
from .operators import WM_OT_numeric_input, WM_OT_numeric_input_key
from .preferences import OnScreenNumpadPreferences
from .utils.ui_utils import CopyTextToClipboardOperator

bl_info = {
    "name": "On-Screen Numpad",
    "author": "Pluglug",
    "version": (1, 1, 0),
    "blender": (4, 2, 0),
    "location": "Numeric Property Fields",
    "description": "No need to leave the mouse to enter numbers!",
    "warning": "",
    "wiki_url": "",
    "category": "User Interface",
}

classes = [
    OnScreenNumpadPreferences,
    WM_OT_numeric_input,
    WM_OT_numeric_input_key,
    CopyTextToClipboardOperator,
]

cleanup_on_reload()
cleanup_menu_on_reload()


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    register_menu()
    register_keymaps()

    from .addon import init_addon

    init_addon()


def unregister():
    unregister_keymaps()
    unregister_menu()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
