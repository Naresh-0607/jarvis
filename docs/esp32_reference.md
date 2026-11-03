# ESP32 Hardware & Pinout Reference Guide

Official reference guide for ESP32 DevKit V1 (30-pin and 38-pin modules) used for grounded hardware mentoring and safety checks.

---

## 1. Electrical Specifications & Voltage Rules

- **Logic Voltage**: **3.3 V**
  - **CRITICAL**: ESP32 GPIO pins are **NOT 5V tolerant**. Connecting a 5V signal directly to any GPIO pin can permanently damage the chip. Always use a logic level converter or resistor voltage divider for 5V signals.
- **Power Inputs**:
  - **Micro-USB / USB-C**: 5 V input, stepped down to 3.3 V by the onboard linear regulator (AMS1117 or similar).
  - **VIN / 5V Pin**: 5 V input to power the onboard regulator.
  - **3V3 Pin**: 3.3 V output from onboard regulator (can supply up to ~500-600 mA), or regulated 3.3 V power input.
- **Current Limits**:
  - Recommended max current per GPIO: **12 mA**.
  - Absolute maximum current per GPIO: **40 mA**.
  - Always use a current-limiting resistor (e.g., 220 Ω to 1 kΩ) when connecting LEDs to GPIO pins.

---

## 2. Pinout & GPIO Classifications

### Built-in Peripherals
- **Onboard LED**: **GPIO 2** (active HIGH on most dev modules; active LOW on a few variants).

### Input-Only Pins (GPI)
- **Pins**: **GPIO 34, GPIO 35, GPIO 36 (VP), GPIO 39 (VN)**
- **Rules**:
  - These pins can **ONLY** be used as inputs. They cannot output signals (no PWM, no HIGH/LOW output).
  - They do **NOT** have software-configurable internal pull-up or pull-down resistors. An external resistor is required if reading a switch or button.

### Strapping Pins (Boot Behavior)
- **GPIO 0**: Controls bootloader mode. Must be HIGH at power-up for normal sketch execution; pulled LOW to enter flashing/download mode.
- **GPIO 2**: Must be left floating or pulled LOW during flashing.
- **GPIO 12 (MTDI)**: Boot fails if pulled HIGH during power-up (selects flash voltage).
- **GPIO 15 (MTDO)**: Boot fails if pulled HIGH during power-up.

### ADC (Analog-to-Digital Converter)
- **Resolution**: 12-bit (values from 0 to 4095), measuring 0 V to ~3.3 V (with 11 dB attenuation).
- **ADC1 (Safe with Wi-Fi)**: **GPIO 32, 33, 34, 35, 36, 39**
  - Use ADC1 pins for analog sensors (potentiometers, light sensors, thermistors).
- **ADC2 (Restricted)**: **GPIO 0, 2, 4, 12, 13, 14, 15, 25, 26, 27**
  - **CRITICAL**: ADC2 cannot be used when Wi-Fi is enabled. The Wi-Fi driver takes exclusive control of ADC2.

### DAC (Digital-to-Analog Converter)
- True 8-bit analog voltage output (0 to 3.3 V):
  - **DAC1**: **GPIO 25**
  - **DAC2**: **GPIO 26**

### Default Communication Busses
- **I2C**:
  - **SDA**: **GPIO 21**
  - **SCL**: **GPIO 22**
- **SPI (VSPI default)**:
  - **MOSI**: **GPIO 23**
  - **MISO**: **GPIO 19**
  - **SCK**: **GPIO 18**
  - **CS / SS**: **GPIO 5**
- **UART (Serial)**:
  - **UART0 (USB Serial)**: TX = **GPIO 1**, RX = **GPIO 3** (used for Serial Monitor and programming).
  - **UART2 (Available)**: TX = **GPIO 17**, RX = **GPIO 16**.

---

## 3. Hardware Safety Checklist

Before instructing the user to make or change any physical connections:
1. **Disconnect Power**: Unplug the USB cable before touching or inserting wires on the breadboard.
2. **Polarity Check**: Verify GND (black/blue wire) is connected to the ground rail and power (red wire) connects to the correct voltage rail (3.3 V or 5 V).
3. **Current Limiting**: Ensure LEDs have a resistor (220 Ω minimum) in series.
4. **Sensor Voltage**: Confirm if the sensor requires 3.3 V or 5 V before connecting VCC.
