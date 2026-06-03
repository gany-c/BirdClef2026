"""
Simple keep-awake script for long Kaggle/BirdCLEF training runs.

Features:
- Prevents system sleep
- Prevents browser/session inactivity
- Simulates tiny keyboard + mouse activity
- Safe fail-stop with Ctrl+C

Works on:
- Windows
- macOS
- Linux (with GUI)

Install:
    pip install pyautogui

Run:
    python keep_awake.py
"""

import time
import random
import pyautogui

# Safety: move mouse to top-left corner to instantly stop script
pyautogui.FAILSAFE = True

# Seconds between actions
INTERVAL = 30

print("Keep-awake script started.")
print("Press CTRL+C to stop.")
print("Move mouse to top-left corner for emergency stop.\n")

try:
    while True:
        # Tiny mouse movement
        move_x = random.randint(-5, 5)
        move_y = random.randint(-5, 5)

        pyautogui.moveRel(move_x, move_y, duration=0.2)

        # Press and release Shift key
        pyautogui.press("shift")

        # Optional tiny scroll
        pyautogui.scroll(random.randint(-1, 1))

        current_time = time.strftime("%H:%M:%S")
        print(f"[{current_time}] Activity sent.")

        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("\nStopped keep-awake script.")
