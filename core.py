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

# 許可する演算子
ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: lambda x, y: float(x) ** float(y),  # 複素数を避けるため独自実装
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

# 許可する数学関数と定数
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
    """プロパティ情報を格納するデータクラス"""

    ptr: Any  # PointerRNA
    prop: Any  # Property definition
    prop_index: int = -1  # Vector要素番号
    sub_path: str = ""  # ID からの相対パス
    id_owner: Any = None  # 最終的に書き込む ID オブジェクト

    def get_display_path(self) -> str:
        """UI表示用のパスを生成"""
        if not self.id_owner:
            return "Unknown"

        try:
            # UIコンテキストオブジェクトの場合の特別処理
            if hasattr(self.id_owner, "bl_rna"):
                class_name = self.id_owner.bl_rna.identifier

                # Screenオブジェクトの場合
                if class_name == "Screen":
                    base_path = "bpy.context.screen"
                # Windowオブジェクトの場合
                elif class_name == "Window":
                    base_path = "bpy.context.window"
                # Sceneオブジェクトの場合
                elif class_name == "Scene":
                    base_path = f'bpy.data.scenes["{self.id_owner.name}"]'
                # Objectオブジェクトの場合
                elif class_name == "Object":
                    base_path = f'bpy.data.objects["{self.id_owner.name}"]'
                # その他のデータブロックの場合
                else:
                    # 従来の処理
                    id_path = f'bpy.data.{self.id_owner.__class__.__name__.lower()}s["{self.id_owner.name}"]'
                    base_path = id_path
            else:
                # bl_rnaがない場合の fallback
                base_path = str(self.id_owner)

            # ネストしたパスの場合、sub_pathは空なので直接プロパティ名を追加
            if self.sub_path:
                full_path = f"{base_path}{self.sub_path}.{self.prop.identifier}"
            else:
                # UIコンテキストプロパティの場合、ネストしたパス全体を再構築する必要がある
                # ただし、今回の実装では簡略化してプロパティ名のみ表示
                full_path = f"{base_path}.{self.prop.identifier}"

            # UIコンテキストオブジェクトの場合の特別処理
            if hasattr(self.id_owner, "bl_rna"):
                class_name = self.id_owner.bl_rna.identifier

                # Screenオブジェクトの場合
                if class_name == "Screen":
                    base_path = "bpy.context.screen"
                # Windowオブジェクトの場合
                elif class_name == "Window":
                    base_path = "bpy.context.window"
                # Sceneオブジェクトの場合
                elif class_name == "Scene":
                    base_path = f'bpy.data.scenes["{self.id_owner.name}"]'
                # Objectオブジェクトの場合
                elif class_name == "Object":
                    base_path = f'bpy.data.objects["{self.id_owner.name}"]'
                # その他のデータブロックの場合
                else:
                    # 従来の処理
                    id_path = f'bpy.data.{self.id_owner.__class__.__name__.lower()}s["{self.id_owner.name}"]'
                    base_path = id_path
            else:
                # bl_rnaがない場合の fallback
                base_path = str(self.id_owner)

            # ネストしたパスの場合、sub_pathは空なので直接プロパティ名を追加
            if self.sub_path:
                full_path = f"{base_path}{self.sub_path}.{self.prop.identifier}"
            else:
                # UIコンテキストプロパティの場合、ネストしたパス全体を再構築する必要がある
                # ただし、今回の実装では簡略化してプロパティ名のみ表示
                full_path = f"{base_path}.{self.prop.identifier}"

            if self.prop_index != -1:
                full_path += f"[{self.prop_index}]"
            return full_path
        except Exception as e:
            log.warning(f"Failed to generate display path: {e}")
            return "Path generation failed"

    def get_current_value(self) -> Union[int, float, None]:
        """現在のプロパティ値を取得"""
        try:
            # ネストしたパスの場合は ptr を直接使用
            # 単純なパスの場合は sub_path を使って解決
            if self.sub_path and self.ptr == self.id_owner:
                # 単純なパス（従来の処理）
                container = self.id_owner.path_resolve(self.sub_path, False)
            else:
                # ネストしたパスまたは直接アクセス
                container = self.ptr

            prop_value = getattr(container, self.prop.identifier)

            if self.prop_index != -1:
                # ベクタープロパティの個別成分の場合
                if (
                    hasattr(prop_value, "__getitem__")
                    and len(prop_value) > self.prop_index
                ):
                    return prop_value[self.prop_index]
            else:
                # スカラープロパティまたはベクタープロパティ全体
                # Vector型の場合は電卓では処理できない
                if hasattr(prop_value, "__len__") and len(prop_value) > 1:
                    log.warning(
                        f"Vector property detected: {self.prop.identifier}. Calculator supports individual components only."
                    )
                    return None
                # スカラープロパティの場合
                return prop_value
        except Exception as e:
            log.warning(f"Failed to get current value: {e}")

        return None

    def get_property_limits(self) -> tuple[Optional[float], Optional[float]]:
        """プロパティの制限値を取得 (min, max)"""
        try:
            hard_min = getattr(self.prop, "hard_min", None)
            hard_max = getattr(self.prop, "hard_max", None)
            return hard_min, hard_max
        except Exception:
            return None, None

    def is_angle_property(self) -> bool:
        """角度プロパティかどうかチェック"""
        try:
            subtype = getattr(self.prop, "subtype", "")
            prop_name = getattr(self.prop, "identifier", "")

            # subtype が ANGLE の場合
            if subtype == "ANGLE":
                return True

            # rotation関連のプロパティ名の場合（クォータニオンは除外）
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

            # EULERサブタイプの場合（回転オイラー角）
            if subtype == "EULER":
                return True

            return False
        except Exception:
            return False


class SafeExpressionEvaluator:
    """安全な数式評価クラス"""

    def __init__(self):
        self.allowed_names = ALLOWED_MATH.copy()

    def evaluate(self, expression: str) -> Union[int, float]:
        """
        安全に数式を評価

        Args:
            expression: 評価する数式

        Returns:
            評価結果（数値）

        Raises:
            ValueError: 無効な数式や禁止された操作
            TypeError: 予期しない型
        """
        if not expression.strip():
            raise ValueError("Empty expression")

        # Blender風の「.1」→「0.1」自動補完
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
        """ASTノードを安全に評価"""

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
            # 複素数や無限大を排除
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
            # 型チェック
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
            # 複素数を排除
            if isinstance(result, complex):
                raise ValueError("Complex numbers are not supported")
            # 型と有限性チェック
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
    """電卓の状態管理クラス"""

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
        """シングルトンインスタンスを取得"""
        return cls()

    @classmethod
    def set_popup(cls, popup_operator):
        """ポップアップオペレータの参照を設定"""
        if popup_operator is None:
            cls._popup_ref = None
        else:
            cls._popup_ref = weakref.ref(popup_operator)
        log.debug(f"Popup reference set: {cls._popup_ref}")

    @classmethod
    def get_popup(cls):
        """ポップアップオペレータの参照を取得"""
        if cls._popup_ref is None:
            return None
        try:
            popup = cls._popup_ref()
            if popup is None:
                cls._popup_ref = None  # 無効な参照をクリア
            return popup
        except Exception:
            cls._popup_ref = None
            return None

    @classmethod
    def clear_popup(cls):
        """ポップアップ参照をクリア"""
        cls.set_popup(None)

    @classmethod
    def cleanup_on_reload(cls):
        """モジュールリロード時のクリーンアップ"""
        if cls._instance:
            cls.clear_popup()
            cls._instance = None
        log.debug("Calculator state cleaned up for reload")

    def evaluate_expression(self, expression: str) -> Union[int, float]:
        """数式を評価して履歴に追加"""
        result = self.evaluator.evaluate(expression)

        # 履歴に追加（重複は避ける）
        if expression not in self.expression_history:
            self.expression_history.append(expression)
            # 履歴サイズ制限（プリファレンスから取得）
            prefs = get_prefs()
            max_history = prefs.get_effective_history_size() if prefs else 20
            if len(self.expression_history) > max_history:
                self.expression_history = self.expression_history[-max_history:]

        log.debug(f"Evaluated: {expression} = {result}")
        return result

    def detect_property_from_context(self, context) -> bool:
        """コンテキストからプロパティを検出"""
        try:
            # 方法1: bpy.context.property（最も直接的）
            if self._try_context_property():
                return True

            # 方法2: copy_data_path_button（フォールバック）
            log.debug("Context property failed, trying copy_data_path_button")
            if self._try_copy_data_path_button(context):
                return True

            log.warning("Could not detect property from any method")
            return False

        except Exception as e:
            log.error(f"Failed to detect property from context: {e}")
            return False

    def _try_context_property(self) -> bool:
        """bpy.context.propertyを使用してプロパティを直接取得"""
        try:
            prop_info = bpy.context.property
            if not prop_info:
                log.debug("No property context available")
                return False

            # prop_infoは(data_block, data_path, index)のタプル
            data_block, data_path, prop_index = prop_info

            if not data_block or not data_path:
                log.debug("Invalid property context")
                return False

            log.debug(f"Context property: {data_block}, {data_path}, {prop_index}")

            # データパスから最終プロパティ名を取得
            prop_name = data_path.split(".")[-1]

            # 配列アクセスを除去（例: "location[0]" -> "location"）
            if "[" in prop_name:
                prop_name = prop_name.split("[")[0]

            # プロパティの所有者を解決
            if "." in data_path:
                # ネストしたパス（例: "node_tree.nodes['Mix'].inputs[0].default_value"）
                try:
                    prop_owner = data_block.path_resolve(data_path.rsplit(".", 1)[0])
                except:
                    log.debug("Failed to resolve property owner from path")
                    return False
            else:
                # 単純なパス（例: "location"）
                prop_owner = data_block

            # プロパティ定義を取得
            if not hasattr(prop_owner, "bl_rna") or not hasattr(
                prop_owner.bl_rna, "properties"
            ):
                log.debug("Property owner has no bl_rna properties")
                return False

            prop_def = prop_owner.bl_rna.properties.get(prop_name)
            if not prop_def:
                log.debug(f"Property definition not found: {prop_name}")
                return False

            # 数値プロパティかチェック
            if prop_def.type not in {"INT", "FLOAT"}:
                log.debug(f"Property is not numeric: {prop_def.type}")
                return False

            # PropertyInfoを作成
            self.current_property = PropertyInfo(
                ptr=prop_owner,
                prop=prop_def,
                prop_index=prop_index if prop_index != -1 else -1,
                sub_path="",  # 直接アクセスなので空
                id_owner=data_block,
            )

            log.debug(
                f"Successfully resolved property via context: {self.current_property.get_display_path()}"
            )
            return True

        except Exception as e:
            log.debug(f"Failed to resolve property via context: {e}")
            log.debug(f"Failed to resolve property via context: {e}")
            return False

    def _try_copy_data_path_button(self, context) -> bool:
        """copy_data_path_buttonを使用してプロパティパスを取得し、evalで解決"""

    def _try_copy_data_path_button(self, context) -> bool:
        """copy_data_path_buttonを使用してプロパティパスを取得し、evalで解決"""
        try:
            # copy_data_path_buttonを呼び出してクリップボードにパスをコピー
            result = bpy.ops.ui.copy_data_path_button(
                full_path=True
            )  # full_path=Trueで完全パスを取得

            if result != {"FINISHED"}:
                log.debug("copy_data_path_button failed")
                log.debug("copy_data_path_button failed")
                return False

            # クリップボードからパスを取得
            clipboard_text = context.window_manager.clipboard
            if not clipboard_text:
                log.debug("No clipboard content available")
                log.debug("No clipboard content available")
                return False

            log.debug(f"Got clipboard path: {clipboard_text}")

            # パスを直接evalで解決
            return self._resolve_path_by_eval(clipboard_text)
            # パスを直接evalで解決
            return self._resolve_path_by_eval(clipboard_text)

        except Exception as e:
            log.debug(f"Failed to resolve property via copy_data_path_button: {e}")
            log.debug(f"Failed to resolve property via copy_data_path_button: {e}")
            return False

    def _resolve_path_by_eval(self, full_path: str) -> bool:
        """完全パスをevalで解決してプロパティ情報を取得"""

    def _resolve_path_by_eval(self, full_path: str) -> bool:
        """完全パスをevalで解決してプロパティ情報を取得"""
        try:
            # パスの例: "bpy.data.objects['Cube'].location[0]"
            # パスの例: "bpy.data.objects['Cube'].location[0]"
            # 配列インデックスを抽出
            prop_index = -1
            base_path = full_path

            if "[" in full_path and full_path.endswith("]"):
                bracket_pos = full_path.rfind("[")
                base_path = full_path[:bracket_pos]
                index_str = full_path[bracket_pos + 1 : -1]
                try:
                    prop_index = int(index_str)
                except ValueError:
                    # 文字列インデックスの場合は無視
                    pass

            # プロパティの所有者を取得
            try:
                prop_owner = eval(base_path.rsplit(".", 1)[0])
                prop_name = base_path.split(".")[-1]
            except:
                log.debug(f"Failed to eval property owner from: {base_path}")
                return False

            # プロパティ定義を取得
            if not hasattr(prop_owner, "bl_rna") or not hasattr(
                prop_owner.bl_rna, "properties"
            ):
                log.debug("Property owner has no bl_rna properties")
                return False

            prop_def = prop_owner.bl_rna.properties.get(prop_name)
            prop_def = prop_owner.bl_rna.properties.get(prop_name)
            if not prop_def:
                log.debug(f"Property definition not found: {prop_name}")
                log.debug(f"Property definition not found: {prop_name}")
                return False

            # 数値プロパティかチェック
            if prop_def.type not in {"INT", "FLOAT"}:
                log.debug(f"Property is not numeric: {prop_def.type}")
                log.debug(f"Property is not numeric: {prop_def.type}")
                return False

            # データブロックIDを取得
            id_owner = getattr(prop_owner, "id_data", prop_owner)

            # PropertyInfoを作成
            self.current_property = PropertyInfo(
                ptr=prop_owner,
                prop=prop_def,
                prop_index=prop_index,
                sub_path="",  # 直接アクセスなので空
                id_owner=id_owner,
            )

            log.debug(
                f"Successfully resolved property via eval: {self.current_property.get_display_path()}"
            )
            return True

        except Exception as e:
            log.debug(f"Failed to resolve property via eval: {e}")
            log.debug(f"Failed to resolve property via eval: {e}")
            return False

    def write_value_to_property(self, value: Union[int, float]) -> bool:
        """プロパティに値を書き込み"""
        if not self.current_property:
            log.error("No property info available")
            return False

        try:
            prefs = get_prefs()

            # プロパティ制限のチェック
            if prefs and prefs.should_respect_limits():
                hard_min, hard_max = self.current_property.get_property_limits()
                if hard_min is not None and value < hard_min:
                    value = hard_min
                    log.debug(f"Value clamped to hard_min: {hard_min}")
                elif hard_max is not None and value > hard_max:
                    value = hard_max
                    log.debug(f"Value clamped to hard_max: {hard_max}")

            # 型変換
            if self.current_property.prop.type == "INT":
                value = int(round(value))

            # コンテナを解決
            try:
                # ネストしたパスの場合は ptr を直接使用
                # 単純なパスの場合は sub_path を使って解決
                if (
                    self.current_property.sub_path
                    and self.current_property.ptr == self.current_property.id_owner
                ):
                    # 単純なパス（従来の処理）
                    container = self.current_property.id_owner.path_resolve(
                        self.current_property.sub_path, False
                    )
                else:
                    # ネストしたパスまたは直接アクセス
                    container = self.current_property.ptr
            except Exception:
                log.warning("Path resolution failed, using original pointer")
                container = self.current_property.ptr

            # 値を書き込み
            prop_name = self.current_property.prop.identifier
            if self.current_property.prop_index != -1:
                # ベクタープロパティの場合
                vec = list(getattr(container, prop_name))
                vec[self.current_property.prop_index] = value
                setattr(container, prop_name, vec)
            else:
                # スカラープロパティの場合
                setattr(container, prop_name, value)

            # デプスグラフ更新でUI反映
            bpy.context.evaluated_depsgraph_get().update()

            # Undo履歴にプッシュ（値の変更が成功した場合のみ）
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
        """プロパティの特性に応じて数式を前処理"""
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

        # 角度プロパティかどうかをチェック
        is_angle = self.current_property.is_angle_property()
        log.debug(
            f"Is angle property: {is_angle} (subtype: {getattr(self.current_property.prop, 'subtype', 'N/A')})"
        )

        # 角度プロパティの場合の自動変換
        if is_angle:
            # 数式に明示的にradians()やdegrees()が含まれていない場合
            has_angle_funcs = any(
                func in expression.lower() for func in ["radians", "degrees", "pi"]
            )
            log.debug(f"Expression has angle functions: {has_angle_funcs}")

            if not has_angle_funcs:
                # 度数法として解釈してラジアンに変換
                log.debug(
                    f"Auto-converting degrees to radians for angle property: {expression}"
                )
                return f"radians({expression})"

        return expression


# モジュールリロード時のクリーンアップ
def cleanup_on_reload():
    """モジュールリロード時に呼び出される"""
    CalculatorState.cleanup_on_reload()
