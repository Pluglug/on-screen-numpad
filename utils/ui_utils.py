import textwrap

import bpy
from bpy.app import version as BL_VERSION
from bpy.types import Operator, UILayout

from ..addon import ADDON_PREFIX, ADDON_PREFIX_PY
from ..utils.logging import get_logger


log = get_logger(__name__)


ICON_ENUM_ITEMS = UILayout.bl_rna.functions["prop"].parameters["icon"].enum_items


def _indented_layout(layout, level):
    """
    指定されたレベルでインデントされたレイアウトを作成

    Args:
        layout: 親レイアウト
        level: インデントレベル（1 = 16px）

    Returns:
        インデントされたカラムレイアウト
    """
    indentpx = 16
    if level == 0:
        level = 0.0001  # 0の場合の調整（半分に分割されないように）
    indent = level * indentpx / bpy.context.region.width

    split = layout.split(factor=indent)
    col = split.column()
    col = split.column()
    return col


class CopyTextToClipboard:
    """テキストをクリップボードにコピーするオペレーター"""

    bl_idname = f"{ADDON_PREFIX_PY}.copy_text_to_clipboard"
    bl_label = "Copy to Clipboard"
    bl_description = "Copy text to clipboard"
    bl_options = {"REGISTER"}

    text: bpy.props.StringProperty(
        name="Text", description="Text to copy to clipboard", default=""
    )

    def execute(self, context):
        try:
            context.window_manager.clipboard = self.text
            self.report({"INFO"}, "Text copied to clipboard")
        except Exception as e:
            self.report({"ERROR"}, f"Failed to copy text: {str(e)}")
            return {"CANCELLED"}

        return {"FINISHED"}


CopyTextToClipboardOperator = type(
    f"{ADDON_PREFIX}_OT_copy_text_to_clipboard",
    (CopyTextToClipboard, Operator),
    {},
)


def ic(icon):
    if not icon:
        return icon

    if icon in ICON_ENUM_ITEMS:
        return icon

    if icon.startswith("SEQUENCE_COLOR_"):  # 4.4+
        return icon.replace("SEQUENCE_COLOR_", "STRIP_COLOR_")

    ICON_ALTERNATIVES = {
        "GREASEPENCIL_LAYER_GROUP": "TEXT",
        "EVENT_NDOF_BUTTON_1": "ONIONSKIN_ON",
        "KEY_BACKSPACE": "BACK",
    }

    if icon in ICON_ALTERNATIVES:
        alt_icon = ICON_ALTERNATIVES[icon]
        return alt_icon

    log.warning(f"Icon not found: {icon}")
    return "BLENDER"


def ic_rb(value):
    return ic("RADIOBUT_ON" if value else "RADIOBUT_OFF")


def ic_cb(value):
    return ic("CHECKBOX_HLT" if value else "CHECKBOX_DEHLT")


def ic_fb(value):
    return ic("SOLO_ON" if value else "SOLO_OFF")


def ic_eye(value):
    return ic("HIDE_OFF" if value else "HIDE_ON")


def ui_prop(layout, data, property, **kwargs):
    if BL_VERSION < (4, 1, 0) and "placeholder" in kwargs:
        del kwargs["placeholder"]

    layout.prop(data, property, **kwargs)


def ui_multiline_text(
    layout,
    text,
    icon=None,
    align="LEFT",
    wrap_width=50,
    text_color="PRIMARY",
    indent=0,
    spacing=False,
    show_copy_button=False,
):
    """
    Blender UILayout内で改行付きの長文テキストをいい感じに表示するユーティリティ

    Args:
        layout: UILayoutオブジェクト
        text: 表示するテキスト（改行文字\nを含む可能性がある）
        icon: 最初の行に表示するアイコン（オプション）
        align: テキストの配置 ('LEFT', 'CENTER', 'RIGHT')
        wrap_width: 自動改行する文字数（デフォルト50）
        text_color: テキストの色テーマ ('PRIMARY', 'SECONDARY', 'WARNING', 'ERROR')
        indent: インデントレベル（1 = 16px）
        spacing: 行間にスペースを追加するか
        show_copy_button: 右下にクリップボードコピーボタンを表示するか
    """
    if not text:
        return

    # テキストを行に分割
    lines = text.split("\n")

    # 各行をさらに指定された幅で自動改行
    formatted_lines = []
    for line in lines:
        if len(line) <= wrap_width:
            formatted_lines.append(line)
        else:
            # textwrapを使用して適切に改行
            wrapped = textwrap.fill(line, width=wrap_width)
            formatted_lines.extend(wrapped.split("\n"))

    # 各行を表示
    for i, line in enumerate(formatted_lines):
        if spacing and i > 0:
            layout.separator(factor=0.1)

        # インデントが指定されている場合、インデントレイアウトを使用
        if indent > 0:
            current_layout = _indented_layout(layout, indent)
        else:
            current_layout = layout

        # 行のレイアウトを作成
        row = current_layout.row()

        # 配置を設定
        if align == "CENTER":
            row.alignment = "CENTER"
        elif align == "RIGHT":
            row.alignment = "RIGHT"
        else:
            row.alignment = "LEFT"

        # テキスト色の設定
        if text_color == "WARNING":
            row.alert = True
        elif text_color == "ERROR":
            row.alert = True

        # 最初の行のみアイコンを表示、それ以降は None
        current_icon = (
            ic(icon) if (i == 0 and icon and ic(icon) != "BLENDER") else "BLANK1"
        )

        # テキストを表示
        if text_color == "SECONDARY":
            # セカンダリテキストの場合は少し暗く表示
            sub_row = row.row()
            sub_row.scale_y = 0.9
            if current_icon:
                sub_row.label(text=line.strip(), icon=current_icon)
            else:
                sub_row.label(text=line.strip())
        else:
            if current_icon:
                row.label(text=line.strip(), icon=current_icon)
            else:
                row.label(text=line.strip())

    # クリップボードコピーボタンを表示
    if show_copy_button:
        copy_row = layout.row()
        copy_row.alignment = "RIGHT"
        copy_row.scale_y = 0.8
        copy_row.scale_x = 0.8

        copy_op = copy_row.operator(
            CopyTextToClipboardOperator.bl_idname, text="", icon="COPYDOWN"
        )
        copy_op.text = text


def ui_text_block(
    layout,
    title,
    text,
    icon=None,
    collapsible=False,
    default_closed=False,
    panel_id=None,
    **text_kwargs,
):
    """
    タイトル付きのテキストブロックを表示するユーティリティ

    Args:
        layout: UILayoutオブジェクト
        title: ブロックのタイトル
        text: 表示するテキスト
        icon: タイトルのアイコン
        collapsible: 折りたたみ可能にするか
        default_closed: デフォルトで閉じているか（Blenderのpanel仕様に合わせて名前変更）
        panel_id: 折りたたみ状態を保存するためのパネルID（collapsible=Trueの場合必須）
        **text_kwargs: ui_multiline_textに渡される追加引数
    """
    if collapsible:
        # 折りたたみ可能なパネル
        if not panel_id:
            # デフォルトのパネルIDを生成
            panel_id = (
                f"TEXT_BLOCK_{title.replace(' ', '_').upper()}"
                if title
                else "TEXT_BLOCK_DEFAULT"
            )

        # Blenderのpanel()メソッドを使用
        header_layout, body_layout = layout.panel(
            panel_id, default_closed=default_closed
        )

        # ヘッダーにタイトルを配置
        if title:
            if icon:
                header_layout.label(text=title, icon=ic(icon))
            else:
                header_layout.label(text=title)

        # ボディが存在する場合（パネルが開いている場合）のみテキストを表示
        if body_layout is not None:
            ui_multiline_text(body_layout, text, **text_kwargs)
    else:
        # 通常のボックス
        box = layout.box()
        col = box.column()

        # タイトル
        if title:
            title_row = col.row()
            if icon:
                title_row.label(text=title, icon=ic(icon))
            else:
                title_row.label(text=title, icon=ic("BLANK1"))
            col.separator(factor=0.5)

        # テキスト内容
        ui_multiline_text(col, text, **text_kwargs)


def ui_help_text(layout, text, icon="INFO", show_copy_button=True, **kwargs):
    """
    ヘルプテキスト用のプリセット
    """
    ui_multiline_text(
        layout,
        text,
        icon=icon,
        text_color="SECONDARY",
        indent=2,
        show_copy_button=show_copy_button,
        **kwargs,
    )


def ui_warning_text(layout, text, icon="ERROR", show_copy_button=False, **kwargs):
    """
    警告テキスト用のプリセット
    """
    ui_multiline_text(
        layout,
        text,
        icon=icon,
        text_color="WARNING",
        show_copy_button=show_copy_button,
        **kwargs,
    )


def ui_error_text(layout, text, icon="CANCEL", show_copy_button=False, **kwargs):
    """
    エラーテキスト用のプリセット
    """
    ui_multiline_text(
        layout,
        text,
        icon=icon,
        text_color="ERROR",
        show_copy_button=show_copy_button,
        **kwargs,
    )
