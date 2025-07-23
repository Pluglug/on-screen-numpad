# pyright: reportInvalidTypeForm=false
from bpy.types import UI_MT_button_context_menu

from .utils.logging import get_logger
from .utils.ui_utils import ic

log = get_logger(__name__)


def draw_numpad_entry(self, context):
    ptr = getattr(context, "button_pointer", None)
    prop = getattr(context, "button_prop", None)

    if ptr and prop and prop.type in {"INT", "FLOAT"}:
        self.layout.separator()
        op = self.layout.operator(
            "wm.numeric_input", text="On-Screen Numpad", icon=ic("TOPBAR")
        )
    else:
        log.debug("No numeric property found")


def register_menu():
    try:
        UI_MT_button_context_menu.append(draw_numpad_entry)
        log.debug("Calculator menu integration registered")
    except Exception as e:
        log.error(f"Failed to register calculator menu: {e}")


def unregister_menu():
    try:
        UI_MT_button_context_menu.remove(draw_numpad_entry)
        log.debug("Calculator menu integration unregistered")
    except Exception as e:
        log.warning(f"Failed to unregister calculator menu: {e}")


def cleanup_menu_on_reload():
    try:
        unregister_menu()
    except:
        pass
