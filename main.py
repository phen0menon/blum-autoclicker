import logging
import random
import time
from typing import Any, List, Optional, TypedDict

import cv2
import keyboard
import mouse
import numpy as np
import pygetwindow as gw
from mss import mss
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)


class Window(TypedDict):
    top: int
    left: int
    width: int
    height: int


def get_point_center(x1: int, y1: int, x2: int, y2: int) -> float:
    return (x1 + x2) / 2, (y1 + y2) / 2


def get_window() -> Optional[Window]:
    try:
        windows = gw.getWindowsWithTitle("TelegramDesktop")

        if not windows:
            raise Exception("Window not found, did you open BLUM app?")

        window = windows[0]

        if not window.isActive:
            window.minimize()
            window.restore()
        return {
            "height": window.height,
            "left": window.left,
            "top": window.top,
            "width": window.width,
        }
    except Exception as ee:
        logging.error("Error getting window: %s", ee)
        window = None


class Runner:
    def __init__(self):
        self.cancelled = True
        self.clicks = 0
        self.init_keybindings()

    def init_keybindings(self):
        keyboard.on_press(self.handle_keyboard_press)

    def handle_keyboard_press(self, event):
        if event.name == "l":
            self.cancelled = True
        elif event.name == "k":
            self.cancelled = False

    def detect_figure_and_click(self, detected: List[Any], window: Window) -> None:
        min_threshold_y = window["top"] + 100
        min_threshold_x = window["left"] + 20
        max_threshold_y = window["top"] + window["height"] - 60
        max_threshold_x = window["left"] + window["width"] - 20

        for result in detected:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                title = box.cls.item()
                # print(f"[title] - {box.cls.item()} [conf] - {box.conf.item()}")

                # ignore bombs
                if title == 3:
                    continue

                object_clickable_shape = None

                if title:
                    center_x, center_y = get_point_center(x1, y1, x2, y2)
                    object_clickable_shape = (center_x, center_y - 10)

                if object_clickable_shape:
                    x = object_clickable_shape[0] + window["left"]
                    y = object_clickable_shape[1] + window["top"]

                    if y > min_threshold_y:
                        mouse.move(x, y, absolute=True)
                        mouse.click(button=mouse.LEFT)
                        time.sleep(0.01)
                        object_clickable_shape = None

                    self.clicks += 1

    def grab_screenshot(self, window: Window):
        with mss() as sct:
            img = sct.grab(
                {
                    "left": window["left"],
                    "top": window["top"],
                    "width": window["width"],
                    "height": window["height"],
                }
            )
            screenshot = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            return screenshot

    def find_replay_button(self, screenshot: np.ndarray, window: Window):
        APPROX_BOTTOM_REPLAY_POS = 200

        white_color = np.array([255, 255, 255])
        mask = cv2.inRange(
            screenshot[-APPROX_BOTTOM_REPLAY_POS:, :], white_color, white_color
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > window["width"] // 2:
                mouse.move(
                    window["left"] + window["width"] // 2,
                    window["top"]
                    + window["height"]
                    - APPROX_BOTTOM_REPLAY_POS
                    + y
                    + h // 2,
                    absolute=True,
                )
                mouse.click(button=mouse.LEFT)
                return True
        return False

    def run(self):
        logging.info("Loading pretrained model...")
        self.model = YOLO("best.pt")
        logging.info("Model loaded")
        logging.info("Ready: Press 'k' to start, 'l' to stop")

        while True:
            if self.cancelled:
                time.sleep(0.1)
                continue

            window = get_window()

            if not window:
                time.sleep(1)
                continue

            try:
                screenshot = self.grab_screenshot(window)
                
                # autoreplay feature
                if self.find_replay_button(screenshot, window):
                    continue

                detected: List[Any] = self.model(screenshot)
                self.detect_figure_and_click(detected, window)
                time.sleep(0.006)
            except Exception as e:
                logging.exception("Error: %s", e)
                continue


if __name__ == "__main__":
    runner = Runner()
    runner.run()
