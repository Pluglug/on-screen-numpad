# On-Screen Numpad

**No need to leave the mouse to enter numbers!**

![Blender](https://img.shields.io/badge/Blender-4.2%2B-orange)
![Version](https://img.shields.io/badge/Version-1.1.0-blue)
![License](https://img.shields.io/badge/License-GPL--3.0--or--later-green)

An on-screen numpad addon for Blender's numeric property input. Simply press a hotkey while your mouse cursor is over a property field to display a calculator-enabled numeric input dialog.

![osn_demo](https://github.com/user-attachments/assets/145214a0-2b3a-41ac-ad94-45d5dd167078)

## Features

**🖱️ Mouse-Optimized**
- Enter numbers without leaving the mouse
- Works with almost all numeric properties in Blender (Transform values, node settings, light/camera settings, etc.)

**🧮 Calculator Functions**
- Arithmetic, power operations, calculations with parentheses
- Math functions (`sin`, `cos`, `tan`, `sqrt`, `log`, `abs`, etc.)
- Math constants (`pi`, `e`, `tau`)

```
50*2.54         → 127    (inch to cm conversion)
2*pi*5          → 31.416 (circumference of radius 5)
sqrt(2)         → 1.414  (diagonal ratio)
```

**📐 Automatic Angle Conversion**
- When enabled, numeric inputs are automatically treated as degrees and converted to radians
- You can also use explicit conversion functions for precise control

```
// When Auto Angle Conversion is enabled
360/4           → 1.571  (auto conversion)
degrees(tau/4)  → 1.571  (explicit degree input)

// Manual conversion (when disabled)
radians(360/4)  → 1.571  (explicit radian conversion)
tau/4           → 1.571  (direct radian value)
```

**📝 Convenient Features**
- History function, use current values, automatic property limit enforcement
- Property path display, keypad layout switching

<img width="487" height="1420" alt="full ui" src="https://github.com/user-attachments/assets/387e6e15-324f-43e8-a7c9-5de258e997f0" />

## Settings

**Display Settings**
- Phone Keypad Layout: Phone-style keypad layout (1-2-3 at top)
- Show Property Path: Display property paths
- Show Function Buttons: Display math function buttons
- Show History: Display calculation history
- Dialog Width: Dialog width (150-600px)
- Decimal Places: Number of decimal places

**Calculation Settings**
- Use Current Value: Use current value as initial value
- Respect Property Limits: Automatically apply property limits
- Auto Angle Conversion: Automatic conversion for angle properties
- History Size: Number of saved history items (1-100)

**Hotkey**
- Default: `Ctrl + Right Click` (customizable)


## License

[GPL-3.0-or-later](LICENSE)

This project is licensed under the GNU General Public License v3.0 or later.

## Contributing

For bug reports and feature requests: [Issues](https://github.com/Pluglug/on-screen-numpad/issues)  
For code improvements: [Pull Requests](https://github.com/Pluglug/on-screen-numpad/pulls)

