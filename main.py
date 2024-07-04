import keyboard
from ultralytics import YOLO
import cv2
import time
import numpy as np
import mouse
import random
import pygetwindow as gw
from mss import mss


cancelled = True
print("Loading model...")
model = YOLO("best.pt")
print("Model loaded")

clicks = 0


def salon_centrifuga(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2


def click(detected, l, t, size, width):
    global clicks
    click_position = None

    threshold_y = t + 100
    threshold_x = l + 20
    max_theshold_y = t + size - 60

    for result in detected:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            title = box.cls.item()
            # print(f"[title] - {box.cls.item()} [conf] - {box.conf.item()}")

            # TODO: maybe add a condition to check if the object is in the region to make it fault respective

            if title == 3 and random.random() > 0.25:
                continue

            if title:
                center_x, center_y = salon_centrifuga(x1, y1, x2, y2)
                object_clickable_shape = (center_x, center_y - 10)

            if object_clickable_shape:
                x = object_clickable_shape[0] + l
                y = object_clickable_shape[1] + t

                if y > threshold_y:
                    mouse.move(x, y, absolute=True)
                    mouse.click(button=mouse.LEFT)
                    time.sleep(0.01)
                    object_clickable_shape = None

                # Fault-respective implementation !! to just click on a HOLE you dumb ass
                if clicks % 4 == 0:
                    # random click
                    x = l + np.random.randint(width)
                    x = max(x, threshold_x)
                    y = t + np.random.randint(size)
                    y = min(max(y, threshold_y), max_theshold_y)
                    mouse.move(x, y, absolute=True)
                    mouse.click(button=mouse.LEFT)
                clicks += 1


def on_escape(event):
    global cancelled
    if event.name == "l":
        cancelled = True
    if event.name == "k":
        cancelled = False


keyboard.on_press(on_escape)


def main():
    print("Running")
    print("Press 'k' to start, 'l' to stop")

    global cancelled

    while True:
        if cancelled:
            time.sleep(0.1)
            continue
        window_title = "TelegramDesktop"

        try:
            window = gw.getWindowsWithTitle(window_title)[0]
            if not window.isActive:
                window.minimize()
                window.restore()
        except Exception as ee:
            print(ee)
            window = None

        try:
            with mss() as sct:
                monitor = {
                    "left": window.left,
                    "top": window.top,
                    "width": window.width,
                    "height": window.height,
                }
                img = sct.grab(monitor)
                screenshot = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                detected = model(screenshot)
                click(detected, window.left, window.top, window.height, window.width)
                time.sleep(0.006)
        except Exception as e:
            print(e)
            continue


main()
