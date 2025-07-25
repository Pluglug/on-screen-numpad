# pyright: reportInvalidTypeForm=false
import bpy
from bpy.types import Operator
from bpy.props import StringProperty, BoolProperty

from .addon import get_prefs
from .utils.logging import get_logger
from .utils.ui_utils import ic, ui_text_block
from .core import CalculatorState

log = get_logger(__name__)


class WM_OT_numeric_input(Operator):
    """Calculator interface for numeric properties"""

    bl_idname = "wm.numeric_input"
    bl_label = "On-Screen Numpad"
    bl_description = "Numpad interface for numeric properties"

    expr: StringProperty(default="")
    initial_value_set: BoolProperty(default=False)

    @classmethod
    def poll(cls, context):
        """Check if operator can be executed"""
        calculator = CalculatorState.get_instance()

        if calculator.get_popup():
            return True

        if calculator.is_numeric_property_available(context):
            return True

        return False

    def invoke(self, context, event):
        """Display calculator dialog"""
        log.debug("WM_OT_numeric_input invoked")
        calculator = CalculatorState.get_instance()

        # Set property information
        if not calculator.detect_property_from_context(context):
            self.report({"ERROR"}, "Unsupported property")
            return {"CANCELLED"}

        # Check property type (type validation for hotkey invocation)
        if (
            not calculator.current_property
            or calculator.current_property.prop.type not in {"INT", "FLOAT"}
        ):
            self.report(
                {"ERROR"},
                "Numpad can only be used with numeric properties (INT/FLOAT)",
            )
            return {"CANCELLED"}

        # Error for vector properties as a whole
        current_value = calculator.current_property.get_current_value()
        if current_value is None:
            self.report(
                {"ERROR"},
                "Numpad can only be used with individual numeric values, not vector properties",
            )
            return {"CANCELLED"}

        # Set calculator reference
        calculator.set_popup(self)

        # Get preferences
        prefs = get_prefs()

        # Check if current value should be displayed
        if prefs and prefs.should_use_current_value():
            current_value = calculator.current_property.get_current_value()
            if (
                current_value is not None
                and calculator.current_property.is_angle_property()
                and prefs.should_convert_angles()
            ):
                # Convert radians to degrees
                import math

                current_value = math.degrees(current_value)

            if current_value is not None:
                # Format with decimal places from preferences
                if prefs:
                    self.expr = prefs.format_result(current_value)
                else:
                    self.expr = str(current_value)
                # Record that initial value was set
                self.initial_value_set = True
            else:
                self.expr = ""
                self.initial_value_set = False
        else:
            self.expr = ""
            self.initial_value_set = False

        log.debug(
            f"Numpad invoked for: {calculator.current_property.get_display_path()}"
        )

        # Get dialog width from preferences
        dialog_width = prefs.dialog_width if prefs else 300
        return context.window_manager.invoke_props_dialog(self, width=dialog_width)

    def draw(self, context):
        """Draw numpad UI"""
        calculator = CalculatorState.get_instance()
        if not calculator.current_property:
            return

        prefs = get_prefs()
        layout = self.layout
        layout.use_property_split = False
        layout.use_property_decorate = False

        # === Property Information Panel ===
        if calculator.current_property and prefs and prefs.show_property_path:
            # Show property name as title, path as body
            prop_name = calculator.current_property.prop.identifier
            prop_path = calculator.current_property.get_display_path()

            # # Calculate wrap width based on dialog width
            # dialog_width = prefs.dialog_width if prefs else 300
            # # Estimate character count (pixel width / 8px per character)
            # wrap_width = max(30, dialog_width // 12)

            # Add property detail information
            additional_info = []
            if (
                prefs.should_respect_limits()
                or calculator.current_property.get_current_value() is not None
            ):
                # Current value
                current_value = calculator.current_property.get_current_value()
                if current_value is not None:
                    current_str = (
                        prefs.format_result(current_value)
                        if prefs
                        else str(current_value)
                    )
                    additional_info.append(f"Current Value: {current_str}")

                # Property limits
                if prefs.should_respect_limits():
                    hard_min, hard_max = (
                        calculator.current_property.get_property_limits()
                    )
                    if hard_min is not None or hard_max is not None:
                        min_str = str(hard_min) if hard_min is not None else "∞"
                        max_str = str(hard_max) if hard_max is not None else "∞"
                        additional_info.append(f"Range: [{min_str} ~ {max_str}]")

            # Build display text
            display_text = prop_path
            if additional_info:
                display_text += "\n" + "\n".join(additional_info)

            # Display property information using ui_text_block
            ui_text_block(
                layout,
                title=prop_name,
                text=display_text,
                icon="RNA",
                collapsible=True,
                default_closed=True,
                panel_id="calc_property_info",
                show_copy_button=True,
            )

        # === Input Area ===
        input_box = layout.box()
        input_col = input_box.column()

        # Input field (larger)
        expr_row = input_col.row(align=True)
        expr_row.scale_y = 1.4
        expr_row.prop(self, "expr", text="", icon=ic("CONSOLE"), placeholder="0")
        op = expr_row.operator("wm.numeric_input_key", text="", icon=ic("PANEL_CLOSE"))
        op.operation = "CLEAR"

        # Angle conversion notice
        if (
            calculator.current_property
            and calculator.current_property.is_angle_property()
            and prefs
            and prefs.should_convert_angles()
        ):
            angle_row = input_col.row()
            angle_row.scale_y = 0.7
            angle_row.alignment = "CENTER"
            angle_row.label(
                text="Degree inputs are automatically converted to radians",
                icon=ic("INFO"),
            )

        # === Numeric Keypad ===
        self._draw_numpad(input_box)

        # === Function Palette ===
        if prefs and prefs.show_functions:
            self._draw_function_buttons(layout)

        # === History Panel ===
        if prefs and prefs.show_history and calculator.expression_history:
            self._draw_history_panel(layout, calculator.expression_history)

    def _draw_function_buttons(self, layout):
        """Draw function buttons"""
        header, body = layout.panel("calc_functions", default_closed=True)
        header.label(text="Math Functions", icon=ic("SCRIPTPLUGINS"))

        if body:
            # Categorize function buttons
            func_col = body.column(align=True)
            func_col.scale_y = 0.9

            # Constants
            const_row = func_col.row(align=True)
            for func, display in [
                ("pi", "π"),
                ("e", "e"),
                ("tau", "τ"),
            ]:
                op = const_row.operator("wm.numeric_input_key", text=display)
                op.operation = "FUNCTION"
                op.value = func

            # Trigonometric functions
            trig_row1 = func_col.row(align=True)
            for func, display in [
                ("sin", "sin"),
                ("cos", "cos"),
                ("tan", "tan"),
                ("atan2", "atan2"),
            ]:
                op = trig_row1.operator("wm.numeric_input_key", text=display)
                op.operation = "FUNCTION"
                op.value = func

            # Inverse trigonometric functions
            trig_row2 = func_col.row(align=True)
            for func, display in [
                ("asin", "asin"),
                ("acos", "acos"),
                ("atan", "atan"),
            ]:
                op = trig_row2.operator("wm.numeric_input_key", text=display)
                op.operation = "FUNCTION"
                op.value = func

            # Angle conversion
            angle_row = func_col.row(align=True)
            for func, display in [
                ("radians", "rad"),
                ("degrees", "deg"),
            ]:
                op = angle_row.operator("wm.numeric_input_key", text=display)
                op.operation = "FUNCTION"
                op.value = func

            # Basic functions
            basic_row1 = func_col.row(align=True)
            for func, display in [
                ("sqrt", "√"),
                ("abs", "abs"),
                ("min", "min"),
                ("max", "max"),
            ]:
                op = basic_row1.operator("wm.numeric_input_key", text=display)
                op.operation = "FUNCTION"
                op.value = func

            # Logarithmic and exponential functions
            log_row = func_col.row(align=True)
            for func, display in [
                ("log", "ln"),
                ("log10", "log"),
                ("exp", "exp"),
            ]:
                op = log_row.operator("wm.numeric_input_key", text=display)
                op.operation = "FUNCTION"
                op.value = func

    def _draw_numpad(self, layout):
        """Draw numpad layout"""
        BUTTON_SCALE_Y = 1.8
        BUTTON_SCALE_X = 0.5
        COMMON_SCALE_Y = 1.1

        prefs = get_prefs()

        num_box = layout.box()

        # # Clear button (top row)
        # clear_row = num_box.row(align=True)
        # clear_row.scale_y = COMMON_SCALE_Y
        # clear_op = clear_row.operator(
        #     "wm.numeric_input_key", text="Clear", icon=ic("CANCEL")
        # )
        # clear_op.operation = "CLEAR"

        # Main keypad layout
        main_row = num_box.row(align=False)

        # Left side: number keypad (3x3 grid)
        numbers_col = main_row.column(align=True)
        numbers_col.scale_y = BUTTON_SCALE_Y
        numbers_col.scale_x = BUTTON_SCALE_X

        # Switch layout between phone and calculator style
        phone_layout = prefs.phone_keypad_layout if prefs else False

        if phone_layout:
            # Phone layout (1-2-3 on top)
            number_rows = [
                ["1", "2", "3"],
                ["4", "5", "6"],
                ["7", "8", "9"],
            ]
        else:
            # Calculator layout (7-8-9 on top)
            number_rows = [
                ["7", "8", "9"],
                ["4", "5", "6"],
                ["1", "2", "3"],
            ]

        for row_numbers in number_rows:
            row = numbers_col.row(align=True)
            for num in row_numbers:
                op = row.operator("wm.numeric_input_key", text=num)
                op.operation = "INPUT"
                op.value = num

        # Bottom row: 0, dot, backspace
        bottom_row = numbers_col.row(align=True)

        # Determine order of 0 and dot based on phone/calculator layout
        keys = [".", "0"] if phone_layout else ["0", "."]
        for key in keys:
            op = bottom_row.operator("wm.numeric_input_key", text=key)
            op.operation = "INPUT"
            op.value = key

        back_op = bottom_row.operator("wm.numeric_input_key", text="⌫")
        back_op.operation = "BACKSPACE"

        # Right side: arithmetic operations (vertical column)
        operators_col = main_row.column(align=True)
        operators_col.scale_y = BUTTON_SCALE_Y
        operators_col.scale_x = BUTTON_SCALE_X

        arithmetic_ops = [
            ("÷", "/"),
            ("×", "*"),
            ("−", "-"),
            ("+", "+"),
        ]

        for display, value in arithmetic_ops:
            op = operators_col.operator("wm.numeric_input_key", text=display)
            op.operation = "INPUT"
            op.value = value

        # Special operations row
        special_row = num_box.row(align=True)
        special_row.scale_y = COMMON_SCALE_Y

        # Parentheses, power, and sign toggle
        paren_open_op = special_row.operator("wm.numeric_input_key", text="(")
        paren_open_op.operation = "INPUT"
        paren_open_op.value = "("

        paren_close_op = special_row.operator("wm.numeric_input_key", text=")")
        paren_close_op.operation = "INPUT"
        paren_close_op.value = ")"

        power_op = special_row.operator("wm.numeric_input_key", text="^")
        power_op.operation = "INPUT"
        power_op.value = "**"

        negate_op = special_row.operator("wm.numeric_input_key", text="±")
        negate_op.operation = "NEGATE"

    def _draw_history_panel(self, layout, history):
        """Draw history panel"""
        header, body = layout.panel("calc_history", default_closed=True)
        header.label(text="History", icon=ic("TIME"))

        if body:
            # Show recent 5 entries
            recent_history = history[-5:]
            if recent_history:
                history_col = body.column(align=True)
                history_col.scale_y = 0.9

                for expr in recent_history:
                    row = history_col.row()
                    op = row.operator(
                        "wm.numeric_input_key", text=expr, icon=ic("GREASEPENCIL")
                    )
                    op.operation = "HISTORY"
                    op.value = expr
            else:
                empty_row = body.row()
                empty_row.scale_y = 0.7
                empty_row.label(text="No history", icon=ic("INFO"))

    def execute(self, context):
        """Execute calculation and apply to property"""
        log.debug("WM_OT_numeric_input_key executed")
        calculator = CalculatorState.get_instance()

        if not calculator.current_property:
            self.report({"ERROR"}, "No property information available")
            return {"CANCELLED"}

        if not self.expr.strip():
            self.report({"ERROR"}, "Empty expression")
            return {"CANCELLED"}

        try:
            # Preprocess expression (angle conversion, etc.)
            processed_expr = calculator.process_expression_for_property(self.expr)

            # Evaluate expression
            result = calculator.evaluate_expression(processed_expr)

            # Write to property
            if calculator.write_value_to_property(result):
                prefs = get_prefs()
                result_str = prefs.format_result(result) if prefs else str(result)
                self.report({"INFO"}, f"Set to {result_str}")
                calculator.clear_popup()
                return {"FINISHED"}
            else:
                self.report({"ERROR"}, "Failed to write value to property")
                return {"CANCELLED"}

        except ValueError as e:
            self.report({"ERROR"}, f"Invalid expression: {e}")
            return {"CANCELLED"}
        except Exception as e:
            log.error(f"Unexpected error in numpad execution: {e}")
            self.report({"ERROR"}, "Calculation failed")
            return {"CANCELLED"}


class WM_OT_numeric_input_key(Operator):
    """Numpad key input operator"""

    bl_idname = "wm.numeric_input_key"
    bl_label = "Numpad Key"
    bl_description = "Numpad key input"

    operation: StringProperty()
    value: StringProperty()

    def execute(self, context):
        """Execute key operation"""
        calculator = CalculatorState.get_instance()
        popup = calculator.get_popup()

        if not popup:
            self.report({"ERROR"}, "Numpad not running")
            return {"CANCELLED"}

        # Auto-clear detection when initial value is displayed
        should_auto_clear = (
            popup.initial_value_set
            and self.operation in ("INPUT", "FUNCTION")
            and (
                self.value.isdigit()
                or self.value in [".", "("]
                or (self.operation == "FUNCTION" and self.value in ["pi", "e", "tau"])
            )
        )

        if should_auto_clear:
            popup.expr = ""
            popup.initial_value_set = False
            log.debug("Auto-cleared initial value before new input")

        if self.operation == "INPUT":
            popup.expr += self.value
            # Clear initial value flag when arithmetic operators are entered (switch to calculation mode)
            if self.value in ["+", "-", "*", "/", ")", "**", "%"]:
                popup.initial_value_set = False
        elif self.operation == "BACKSPACE":
            popup.expr = popup.expr[:-1]
            # Clear initial value flag when editing starts with backspace
            popup.initial_value_set = False
        elif self.operation == "CLEAR":
            popup.expr = ""
            popup.initial_value_set = False
        elif self.operation == "NEGATE":
            if popup.expr:
                # Wrap current expression in parentheses and negate
                popup.expr = f"-({popup.expr})"
            popup.initial_value_set = False
        elif self.operation == "FUNCTION":
            # Insert function name (add parentheses for arguments)
            if self.value in ["pi", "e", "tau"]:
                popup.expr += self.value
            else:
                # Auto-insert current value in parentheses if current value is set
                if popup.initial_value_set and popup.expr.strip():
                    current_expr = popup.expr
                    popup.expr = f"{self.value}({current_expr})"
                    popup.initial_value_set = False
                else:
                    popup.expr += f"{self.value}("
        elif self.operation == "HISTORY":
            popup.expr = self.value
            popup.initial_value_set = False

        log.debug(
            f"Numpad key operation: {self.operation} = {self.value}, expr: {popup.expr}"
        )
        return {"FINISHED"}
