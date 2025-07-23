# pyright: reportInvalidTypeForm=false
from bpy.props import BoolProperty, IntProperty
from bpy.types import AddonPreferences

from .keymaps import draw_keymap
from .utils.ui_utils import ic


class OnScreenNumpadPreferences(AddonPreferences):

    bl_idname = __package__

    # UI display options
    use_current_value: BoolProperty(
        name="Use Current Value",
        description="Use the current property value in the input field when opening numpad",
        default=True,
    )

    show_property_path: BoolProperty(
        name="Show Property Path",
        description="Display the full property path in the numpad dialog",
        default=True,
    )

    show_history: BoolProperty(
        name="Show History",
        description="Display calculation history in the numpad dialog",
        default=True,
    )

    show_functions: BoolProperty(
        name="Show Function Buttons",
        description="Display mathematical function buttons",
        default=True,
    )

    phone_keypad_layout: BoolProperty(
        name="Phone Keypad Layout",
        description="Use phone-style keypad layout (1-2-3 at top) instead of calculator-style (7-8-9 at top)",
        default=False,
    )

    dialog_width: IntProperty(
        name="Dialog Width",
        description="Width of the numpad dialog",
        default=200,
        min=150,
        max=600,
    )

    decimal_places: IntProperty(
        name="Decimal Places",
        description="Number of decimal places to display in results",
        default=3,
        min=0,
        max=10,
    )

    # Calculation options
    respect_property_limits: BoolProperty(
        name="Respect Property Limits",
        description="Automatically clamp values to property min/max limits",
        default=True,
    )

    auto_angle_conversion: BoolProperty(
        name="Auto Angle Conversion",
        description="Automatically convert degrees to radians for angle properties",
        default=True,
    )

    # History settings
    history_size: IntProperty(
        name="History Size",
        description="Number of expressions to keep in history",
        default=10,
        min=1,
        max=100,
    )

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        setting_box = layout.box()

        setting_box.label(text="Settings", icon=ic("PREFERENCES"))

        col = setting_box.column(heading="Display")
        col.prop(self, "phone_keypad_layout")
        col.prop(self, "show_property_path")
        col.prop(self, "show_functions")
        col.prop(self, "show_history")
        col.prop(self, "dialog_width")
        col.prop(self, "decimal_places")

        col = setting_box.column(heading="Calculation")
        col.prop(self, "use_current_value")
        col.prop(self, "respect_property_limits")
        col.prop(self, "auto_angle_conversion")
        col.prop(self, "history_size")

        keymap_box = layout.box()
        keymap_box.label(text="Hotkeys", icon=ic("MOUSE_MMB"))
        draw_keymap(self, context, keymap_box)

    def get_effective_history_size(self) -> int:
        """Get effective history size"""
        return max(5, min(100, self.history_size))

    def should_use_current_value(self) -> bool:
        return self.use_current_value

    def should_respect_limits(self) -> bool:
        return self.respect_property_limits

    def should_convert_angles(self) -> bool:
        return self.auto_angle_conversion

    def format_result(self, value) -> str:
        """Format result with specified decimal places"""
        # Return as string for None values or Vector types
        if value is None:
            return "N/A"

        # For other non-numeric types (Vector, etc.)
        if not isinstance(value, (int, float)):
            return str(value)

        # For numeric values, format with specified decimal places
        if self.decimal_places == 0:
            return str(int(round(value)))
        else:
            return f"{value:.{self.decimal_places}f}"
