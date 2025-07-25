import ast
import math
import operator as op
import re
import weakref
from dataclasses import dataclass
from typing import Any, Optional, Union

import bpy

from .addon import get_prefs
from .utils.logging import get_logger

log = get_logger(__name__)

# Allowed operators
ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: lambda x, y: float(x)
    ** float(y),  # Custom implementation to avoid complex numbers
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

# Allowed math functions and constants
ALLOWED_MATH = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "abs": abs,
    "min": min,
    "max": max,
    "radians": math.radians,
    "degrees": math.degrees,
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


@dataclass
class PropertyInfo:
    """Data class for storing property information"""

    ptr: Any  # PointerRNA
    prop: Any  # Property definition
    prop_index: int = -1  # Vector element index
    sub_path: str = ""  # Relative path from ID
    id_owner: Any = None  # ID object for final writing

    def get_display_path(self) -> str:
        """Generate path for UI display"""
        if not self.id_owner:
            return "Unknown"

        try:
            # Generate base path based on object type
            if hasattr(self.id_owner, "bl_rna"):
                class_name = self.id_owner.bl_rna.identifier

                # Special handling for UI context objects
                if class_name == "Screen":
                    base_path = "bpy.context.screen"
                elif class_name == "Window":
                    base_path = "bpy.context.window"
                elif class_name == "Scene":
                    base_path = f'bpy.data.scenes["{self.id_owner.name}"]'
                elif class_name == "Object":
                    base_path = f'bpy.data.objects["{self.id_owner.name}"]'
                else:
                    # Default processing for other data blocks
                    id_path = f'bpy.data.{self.id_owner.__class__.__name__.lower()}s["{self.id_owner.name}"]'
                    base_path = id_path
            else:
                # Fallback when bl_rna is not available
                base_path = str(self.id_owner)

            # Build full path
            if self.sub_path:
                full_path = f"{base_path}{self.sub_path}.{self.prop.identifier}"
            else:
                full_path = f"{base_path}.{self.prop.identifier}"

            # Add array index if applicable
            if self.prop_index != -1:
                full_path += f"[{self.prop_index}]"
            return full_path
        except Exception as e:
            log.warning(f"Failed to generate display path: {e}")
            return "Path generation failed"

    def get_current_value(self) -> Union[int, float, None]:
        """Get current property value"""
        try:
            # For nested paths, use ptr directly
            # For simple paths, resolve using sub_path
            if self.sub_path and self.ptr == self.id_owner:
                # Simple path (legacy processing)
                container = self.id_owner.path_resolve(self.sub_path, False)
            else:
                # Nested path or direct access
                container = self.ptr

            prop_value = getattr(container, self.prop.identifier)

            if self.prop_index != -1:
                # Individual component of vector property
                if (
                    hasattr(prop_value, "__getitem__")
                    and len(prop_value) > self.prop_index
                ):
                    return prop_value[self.prop_index]
            else:
                # Scalar property or entire vector property
                # Vector types cannot be processed by calculator
                if hasattr(prop_value, "__len__") and len(prop_value) > 1:
                    log.warning(
                        f"Vector property detected: {self.prop.identifier}. Calculator supports individual components only."
                    )
                    return None
                # Scalar property
                return prop_value
        except Exception as e:
            log.warning(f"Failed to get current value: {e}")

        return None

    def get_property_limits(self) -> tuple[Optional[float], Optional[float]]:
        """Get property limits (min, max)"""
        try:
            hard_min = getattr(self.prop, "hard_min", None)
            hard_max = getattr(self.prop, "hard_max", None)
            return hard_min, hard_max
        except Exception:
            return None, None

    def is_angle_property(self) -> bool:
        """Check if this is an angle property"""
        try:
            subtype = getattr(self.prop, "subtype", "")
            prop_name = getattr(self.prop, "identifier", "")

            # Properties with ANGLE subtype
            if subtype == "ANGLE":
                return True

            # Rotation-related property names (excluding quaternions)
            angle_property_names = [
                "rotation",
                "rotation_euler",
                "rotation_axis_angle",
                "rotation_x",
                "rotation_y",
                "rotation_z",
            ]

            if prop_name in angle_property_names:
                return True

            # EULER subtype (rotation Euler angles)
            if subtype == "EULER":
                return True

            return False
        except Exception:
            return False


class SafeExpressionEvaluator:
    """Safe mathematical expression evaluator"""

    def __init__(self):
        self.allowed_names = ALLOWED_MATH.copy()

    def evaluate(self, expression: str) -> Union[int, float]:
        """
        Safely evaluate mathematical expression

        Args:
            expression: Expression to evaluate

        Returns:
            Evaluation result (numeric)

        Raises:
            ValueError: Invalid expression or forbidden operation
            TypeError: Unexpected type
        """
        if not expression.strip():
            raise ValueError("Empty expression")

        # Blender-style ".1" → "0.1" auto-completion
        original_expr = expression
        expression = re.sub(r"(^|[^0-9])\.([0-9]+)", r"\g<1>0.\2", expression)
        if expression != original_expr:
            log.debug(f"Auto-completed decimal: '{original_expr}' → '{expression}'")

        try:
            node = ast.parse(expression, mode="eval")
            return self._eval_node(node.body)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in expression: {e}")
        except Exception as e:
            raise ValueError(f"Evaluation error: {e}")

    def _eval_node(self, node: ast.AST) -> Union[int, float]:
        """Safely evaluate AST node"""

        if isinstance(node, ast.Constant):  # Python 3.8+
            if isinstance(node.value, (int, float)):
                return node.value
            else:
                raise TypeError(f"Unsupported constant type: {type(node.value)}")

        elif isinstance(node, ast.Num):  # Python < 3.8 compatibility
            return node.n

        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_func = ALLOWED_OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
            result = op_func(left, right)
            # Exclude complex numbers and infinity
            if isinstance(result, complex):
                raise ValueError("Complex numbers are not supported")
            if not isinstance(result, (int, float)) or not (
                float("-inf") < result < float("inf")
            ):
                raise ValueError("Result must be a finite real number")
            return result

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op_func = ALLOWED_OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(
                    f"Unsupported unary operation: {type(node.op).__name__}"
                )
            result = op_func(operand)
            # Type check
            if not isinstance(result, (int, float)) or not (
                float("-inf") < result < float("inf")
            ):
                raise ValueError("Result must be a finite real number")
            return result

        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name not in self.allowed_names:
                raise ValueError(f"Function '{func_name}' is not allowed")

            args = [self._eval_node(arg) for arg in node.args]
            func = self.allowed_names[func_name]
            result = func(*args)
            # Exclude complex numbers
            if isinstance(result, complex):
                raise ValueError("Complex numbers are not supported")
            # Type and finite check
            if not isinstance(result, (int, float)) or not (
                float("-inf") < result < float("inf")
            ):
                raise ValueError("Result must be a finite real number")
            return result

        elif isinstance(node, ast.Name):
            if node.id in self.allowed_names:
                return self.allowed_names[node.id]
            else:
                raise ValueError(f"Name '{node.id}' is not allowed")

        else:
            raise ValueError(f"Unsupported node type: {type(node).__name__}")


class CalculatorState:
    """Calculator state management class"""

    _instance: Optional["CalculatorState"] = None
    _popup_ref: Optional[weakref.ref] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not getattr(self, "_initialized", False):
            self.evaluator = SafeExpressionEvaluator()
            self.current_property: Optional[PropertyInfo] = None
            self.expression_history: list[str] = []
            self._initialized = True

    @classmethod
    def get_instance(cls) -> "CalculatorState":
        """Get singleton instance"""
        return cls()

    @classmethod
    def set_popup(cls, popup_operator):
        """Set popup operator reference"""
        if popup_operator is None:
            cls._popup_ref = None
        else:
            cls._popup_ref = weakref.ref(popup_operator)
        log.debug(f"Popup reference set: {cls._popup_ref}")

    @classmethod
    def get_popup(cls):
        """Get popup operator reference"""
        if cls._popup_ref is None:
            return None
        try:
            popup_ref = cls._popup_ref
            if popup_ref is not None:
                popup = popup_ref()  # type: ignore[operator]
            else:
                popup = None
            if popup is None:
                cls._popup_ref = None  # Clear invalid reference
            return popup
        except Exception:
            cls._popup_ref = None
            return None

    @classmethod
    def clear_popup(cls):
        """Clear popup reference"""
        cls.set_popup(None)
        if cls._instance is not None:
            cls._instance.current_property = None

    @classmethod
    def cleanup_on_reload(cls):
        """Cleanup on module reload"""
        if cls._instance:
            cls.clear_popup()
            cls._instance = None
        log.debug("Calculator state cleaned up for reload")

    def is_numeric_property_available(self, context) -> bool:
        """Return True if *context* is currently pointing at an INT/FLOAT property."""

        # 1) button_pointer / button_prop
        ptr = getattr(context, "button_pointer", None)
        prop = getattr(context, "button_prop", None)
        if ptr and prop and getattr(prop, "type", None) in {"INT", "FLOAT"}:
            return True

        # 2) context.property (forPreferences)
        try:
            prop_info = context.property  # (data_block, data_path, index)
            if prop_info:
                data_block, data_path, _ = prop_info
                if data_block and data_path:
                    prop_name = data_path.split(".")[-1]
                    if "[" in prop_name:
                        prop_name = prop_name.split("[")[0]

                    prop_owner = data_block
                    if "." in data_path:
                        try:
                            prop_owner = data_block.path_resolve(
                                data_path.rsplit(".", 1)[0]
                            )
                        except Exception:
                            prop_owner = None

                    if prop_owner and hasattr(prop_owner, "bl_rna"):
                        prop_def = prop_owner.bl_rna.properties.get(prop_name)
                        if prop_def and prop_def.type in {"INT", "FLOAT"}:
                            return True
        except Exception:
            pass

        try:
            return bpy.ops.ui.copy_data_path_button.poll()
        except Exception:
            return False

    def evaluate_expression(self, expression: str) -> Union[int, float]:
        """Evaluate expression and add to history"""
        result = self.evaluator.evaluate(expression)

        # Add to history (avoid duplicates)
        if expression not in self.expression_history:
            self.expression_history.append(expression)
            # Limit history size (get from preferences)
            prefs = get_prefs()
            max_history = prefs.get_effective_history_size() if prefs else 20
            if len(self.expression_history) > max_history:
                self.expression_history = self.expression_history[-max_history:]

        log.debug(f"Evaluated: {expression} = {result}")
        return result

    def detect_property_from_context(self, context) -> bool:
        """Detect property from context"""
        try:
            # Method 1: bpy.context.property (most direct)
            if self._try_context_property():
                return True

            # Method 2: copy_data_path_button (fallback)
            log.debug("Context property failed, trying copy_data_path_button")
            if self._try_copy_data_path_button(context):
                return True

            log.warning("Could not detect property from any method")
            return False

        except Exception as e:
            log.error(f"Failed to detect property from context: {e}")
            return False

    def _try_context_property(self) -> bool:
        """Get property directly using bpy.context.property"""
        try:
            prop_info = bpy.context.property
            if not prop_info:
                log.debug("No property context available")
                return False

            # prop_info is (data_block, data_path, index) tuple
            data_block, data_path, prop_index = prop_info

            if not data_block or not data_path:
                log.debug("Invalid property context")
                return False

            log.debug(f"Context property: {data_block}, {data_path}, {prop_index}")

            # Get final property name from data path
            prop_name = data_path.split(".")[-1]

            # Remove array access (e.g., "location[0]" -> "location")
            if "[" in prop_name:
                prop_name = prop_name.split("[")[0]

            # Resolve property owner
            if "." in data_path:
                # Nested path (e.g., "node_tree.nodes['Mix'].inputs[0].default_value")
                try:
                    prop_owner = data_block.path_resolve(data_path.rsplit(".", 1)[0])
                except:
                    log.debug("Failed to resolve property owner from path")
                    return False
            else:
                # Simple path (e.g., "location")
                prop_owner = data_block

            # Get property definition
            if not hasattr(prop_owner, "bl_rna") or not hasattr(
                prop_owner.bl_rna, "properties"
            ):
                log.debug("Property owner has no bl_rna properties")
                return False

            prop_def = prop_owner.bl_rna.properties.get(prop_name)
            if not prop_def:
                log.debug(f"Property definition not found: {prop_name}")
                return False

            # Check if numeric property
            if prop_def.type not in {"INT", "FLOAT"}:
                log.debug(f"Property is not numeric: {prop_def.type}")
                return False

            # Create PropertyInfo
            self.current_property = PropertyInfo(
                ptr=prop_owner,
                prop=prop_def,
                prop_index=prop_index if prop_index != -1 else -1,
                sub_path="",  # Empty for direct access
                id_owner=data_block,
            )

            log.debug(
                f"Successfully resolved property via context: {self.current_property.get_display_path()}"
            )
            return True

        except Exception as e:
            log.debug(f"Failed to resolve property via context: {e}")
            return False

    def _parse_blender_path(self, path: str) -> Any:
        """Parse Blender property path

        Args:
            path: Blender property path (e.g., "bpy.data.objects['Cube']")

        Returns:
            The resolved object or None if parsing fails
        """
        if not path or not path.startswith("bpy."):
            return None

        try:
            parts = path.split(".", 2)  # ['bpy', 'data'/'context', ...]
            if len(parts) < 2:
                return None

            # Handle bpy.context paths
            if parts[1] == "context":
                return self._parse_context_path(parts)
            # Handle bpy.data paths
            elif parts[1] == "data":
                return self._parse_data_path(parts)
            # Other bpy attributes
            else:
                return self._parse_other_path(parts)

        except Exception as e:
            log.debug(f"Error parsing path '{path}': {e}")
            return None

    def _parse_context_path(self, parts: list[str]) -> Any:
        """Parse bpy.context.* paths"""
        if len(parts) == 2:
            return bpy.context

        obj = bpy.context
        for attr in parts[2].split("."):
            obj = self._resolve_attribute(obj, attr)
            if obj is None:
                return None
        return obj

    def _parse_data_path(self, parts: list[str]) -> Any:
        """Parse bpy.data.* paths"""
        if len(parts) == 2:
            return bpy.data

        remaining_path = parts[2]

        # Try string indexed access first (e.g., objects['Cube'])
        match = re.match(r"^(\w+)\[(['\"])(.+?)\2\](.*)$", remaining_path)
        if match:
            collection_name, _, obj_name, rest_path = match.groups()
            collection = getattr(bpy.data, collection_name, None)
            if collection is None:
                return None
            obj = collection.get(obj_name)
        else:
            # Try numeric indexed access (e.g., objects[0])
            match = re.match(r"^(\w+)\[(\d+)\](.*)$", remaining_path)
            if not match:
                return None
            collection_name, index_str, rest_path = match.groups()
            collection = getattr(bpy.data, collection_name, None)
            if collection is None:
                return None
            try:
                obj = collection[int(index_str)]
            except (IndexError, KeyError):
                return None

        if obj is None:
            return None

        # Process remaining path
        if rest_path and rest_path.startswith("."):
            for part in rest_path[1:].split("."):
                obj = self._resolve_attribute(obj, part)
                if obj is None:
                    return None
        return obj

    def _parse_other_path(self, parts: list[str]) -> Any:
        """Parse other bpy.* paths"""
        obj = getattr(bpy, parts[1], None)
        if obj is None or len(parts) == 2:
            return obj

        for attr in parts[2].split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                return None
        return obj

    def _resolve_attribute(self, obj: Any, attr: str) -> Any:
        """Resolve a single attribute with optional index"""
        if "[" not in attr or not attr.endswith("]"):
            return getattr(obj, attr, None)

        # Handle indexed access
        attr_name = attr[: attr.index("[")]
        index_str = attr[attr.index("[") + 1 : -1].strip("'\"")

        obj = getattr(obj, attr_name, None)
        if obj is None:
            return None

        try:
            # Try numeric index
            return obj[int(index_str)]
        except ValueError:
            # String index
            return obj.get(index_str) if hasattr(obj, "get") else obj[index_str]

    def _try_copy_data_path_button(self, context) -> bool:
        """Get property path using copy_data_path_button"""
        try:
            # Call copy_data_path_button to copy path to clipboard
            result = bpy.ops.ui.copy_data_path_button(
                full_path=True
            )  # Get full path with full_path=True

            if result != {"FINISHED"}:
                log.debug("copy_data_path_button failed")
                return False

            # Get path from clipboard
            clipboard_text = context.window_manager.clipboard
            if not clipboard_text:
                log.debug("No clipboard content available")
                return False

            log.debug(f"Got clipboard path: {clipboard_text}")

            # Resolve path
            return self._resolve_path(clipboard_text)

        except Exception as e:
            log.debug(f"Failed to resolve property via copy_data_path_button: {e}")
            return False

    def _resolve_path(self, full_path: str) -> bool:
        """Resolve full path to get property information"""
        try:
            # Path example: "bpy.data.objects['Cube'].location[0]"
            # Extract array index
            prop_index = -1
            base_path = full_path

            if "[" in full_path and full_path.endswith("]"):
                bracket_pos = full_path.rfind("[")
                base_path = full_path[:bracket_pos]
                index_str = full_path[bracket_pos + 1 : -1]
                try:
                    prop_index = int(index_str)
                except ValueError:
                    # Ignore string indices
                    pass

            # Parse property path
            prop_owner = self._parse_blender_path(base_path.rsplit(".", 1)[0])
            if prop_owner is None:
                log.debug(f"Failed to parse property owner from: {base_path}")
                return False
            prop_name = base_path.split(".")[-1]

            # Get property definition
            if not hasattr(prop_owner, "bl_rna") or not hasattr(
                prop_owner.bl_rna, "properties"
            ):
                log.debug("Property owner has no bl_rna properties")
                return False

            prop_def = prop_owner.bl_rna.properties.get(prop_name)
            if not prop_def:
                log.debug(f"Property definition not found: {prop_name}")
                return False

            # Check if numeric property
            if prop_def.type not in {"INT", "FLOAT"}:
                log.debug(f"Property is not numeric: {prop_def.type}")
                return False

            # Get data block ID
            id_owner = getattr(prop_owner, "id_data", prop_owner)

            # Create PropertyInfo
            self.current_property = PropertyInfo(
                ptr=prop_owner,
                prop=prop_def,
                prop_index=prop_index,
                sub_path="",  # Empty for direct access
                id_owner=id_owner,
            )

            log.debug(
                f"Successfully resolved property: {self.current_property.get_display_path()}"
            )
            return True

        except Exception as e:
            log.debug(f"Failed to resolve property: {e}")
            return False

    def write_value_to_property(self, value: Union[int, float]) -> bool:
        """Write value to property"""
        if not self.current_property:
            log.error("No property info available")
            return False

        try:
            prefs = get_prefs()

            # Check property limits
            if prefs and prefs.should_respect_limits():
                hard_min, hard_max = self.current_property.get_property_limits()
                if hard_min is not None and value < hard_min:
                    value = hard_min
                    log.debug(f"Value clamped to hard_min: {hard_min}")
                elif hard_max is not None and value > hard_max:
                    value = hard_max
                    log.debug(f"Value clamped to hard_max: {hard_max}")

            # Type conversion
            if self.current_property.prop.type == "INT":
                value = int(round(value))

            # Resolve container
            try:
                # For nested paths, use ptr directly
                # For simple paths, resolve using sub_path
                if (
                    self.current_property.sub_path
                    and self.current_property.ptr == self.current_property.id_owner
                ):
                    # Simple path (legacy processing)
                    container = self.current_property.id_owner.path_resolve(
                        self.current_property.sub_path, False
                    )
                else:
                    # Nested path or direct access
                    container = self.current_property.ptr
            except Exception:
                log.warning("Path resolution failed, using original pointer")
                container = self.current_property.ptr

            # Write value
            prop_name = self.current_property.prop.identifier
            if self.current_property.prop_index != -1:
                # Vector property
                vec = list(getattr(container, prop_name))
                vec[self.current_property.prop_index] = value
                setattr(container, prop_name, vec)
            else:
                # Scalar property
                setattr(container, prop_name, value)

            # Update depsgraph for UI reflection
            bpy.context.evaluated_depsgraph_get().update()

            # Push to undo history (only when value change succeeds)
            try:
                bpy.ops.ed.undo_push(message=f"Calculator: Set {prop_name} to {value}")
                log.debug(f"Undo pushed for property change: {prop_name} = {value}")
            except Exception as e:
                log.warning(f"Failed to push undo: {e}")

            log.debug(f"Successfully wrote value {value} to property")
            return True

        except Exception as e:
            log.error(f"Failed to write value to property: {e}")
            return False

    def process_expression_for_property(self, expression: str) -> str:
        """Preprocess expression according to property characteristics"""
        if not self.current_property:
            log.debug("No current property for expression processing")
            return expression

        prefs = get_prefs()
        log.debug(
            f"Angle conversion setting: {prefs.should_convert_angles() if prefs else 'No prefs'}"
        )

        if not prefs or not prefs.should_convert_angles():
            log.debug("Angle conversion disabled in preferences")
            return expression

        # Check if angle property
        is_angle = self.current_property.is_angle_property()
        log.debug(
            f"Is angle property: {is_angle} (subtype: {getattr(self.current_property.prop, 'subtype', 'N/A')})"
        )

        # Auto conversion for angle properties
        if is_angle:
            # If expression doesn't explicitly contain radians() or degrees()
            has_angle_funcs = any(
                func in expression.lower() for func in ["radians", "degrees", "pi"]
            )
            log.debug(f"Expression has angle functions: {has_angle_funcs}")

            if not has_angle_funcs:
                # Interpret as degrees and convert to radians
                log.debug(
                    f"Auto-converting degrees to radians for angle property: {expression}"
                )
                return f"radians({expression})"

        return expression


# Cleanup on module reload
def cleanup_on_reload():
    """Called on module reload"""
    CalculatorState.cleanup_on_reload()
