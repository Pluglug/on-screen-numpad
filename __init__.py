from bpy.utils import register_class, unregister_class

from .core import cleanup_on_reload
from .keymaps import register_keymaps, unregister_keymaps
from .menu_integration import cleanup_menu_on_reload, register_menu, unregister_menu
from .operators import WM_OT_numeric_input, WM_OT_numeric_input_key
from .preferences import OnScreenNumpadPreferences
from .utils.ui_utils import OSN_OT_copy_text_to_clipboard

bl_info = {
    "name": "On-Screen Numpad",
    "author": "Pluglug",
    "version": (1, 2, 0),
    "blender": (4, 2, 0),
    "location": "Numeric Property Fields",
    "description": "No need to leave the mouse to enter numbers!",
    "warning": "",
    "wiki_url": "",
    "category": "User Interface",
}

# Version update checklist (when bumping version)
# - [ ] Update bl_info["version"]
# - [ ] Update version in blender_manifest.toml
# - [ ] Update blender_version_max if needed
# - [ ] Update Version badge in README.md

classes = [
    OnScreenNumpadPreferences,
    WM_OT_numeric_input,
    WM_OT_numeric_input_key,
    OSN_OT_copy_text_to_clipboard,
]

cleanup_on_reload()
cleanup_menu_on_reload()


def register():
    for cls in classes:
        register_class(cls)
    register_menu()
    register_keymaps()

    from .addon import init_addon

    init_addon()


def unregister():
    unregister_keymaps()
    unregister_menu()
    for cls in reversed(classes):
        unregister_class(cls)
