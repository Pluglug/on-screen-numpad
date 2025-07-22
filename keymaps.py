import bpy
from rna_keymap_ui import draw_kmi


addon_keymaps = []


def register_keymaps():
    wm = bpy.context.window_manager
    if wm.keyconfigs.addon is None:
        return
    km = wm.keyconfigs.addon.keymaps.new(name="User Interface", space_type="EMPTY")
    kmi = km.keymap_items.new(
        idname="wm.numeric_input", type="RIGHTMOUSE", value="PRESS", ctrl=True
    )
    addon_keymaps.append((km, kmi))


def unregister_keymaps():
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()


def draw_keymap(self, context, layout, level=0):
    if kc := context.window_manager.keyconfigs.user:
        km = kc.keymaps.new(name="User Interface", space_type="EMPTY")
        for kmi in km.keymap_items:
            if kmi.idname == "wm.numeric_input":
                draw_kmi([], kc, km, kmi, layout, level)
