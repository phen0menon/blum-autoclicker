import logging
import random
import time
from typing import Any, Dict, List, Literal, Optional, TypedDict

import cv2
import keyboard
import mouse
import numpy as np
import pygetwindow as gw
from mss import mss
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

Language = Literal["en", "ru"]
Translations = Dict[str, str]

locales: Dict[Language, Translations] = {
    "en": {
        "loading": "Loading pretrained model...",
        "window_not_found": "Window not found, did you open BLUM app?",
        "error_getting_window": "Error getting window",
        "ready": "Ready: Start the Drop Game and then press 'k' to start, 'l' to stop",
        "cold_start": "Model is running first time, may take a few seconds...",
        "stopped": "⏸️  Stopped, press 'k' to start again",
        "running": "🔥 Running, press 'l' to stop",
    },
    "ru": {
        "loading": "Загрузка обученной модели...",
        "window_not_found": "Окно не найдено, вы открыли приложение BLUM?",
        "error_getting_window": "Ошибка получения окна",
        "ready": "Готово: Запустите мини-игру и нажмите 'k' для начала, 'l' для остановки",
        "cold_start": "Модель запускается впервые, это может занять несколько секунд...",
        "stopped": "⏸️  Остановлено, для запуска снова нажмите 'k'",
        "running": "🔥 Запущено, для остановки нажмите 'l'",
    },
}


class Window(TypedDict):
    top: int
    left: int
    width: int
    height: int


def get_point_center(x1: int, y1: int, x2: int, y2: int) -> float:
    return (x1 + x2) / 2, (y1 + y2) / 2


def get_window(locale: Translations) -> Optional[Window]:
    try:
        windows = gw.getWindowsWithTitle("TelegramDesktop")

        if not windows:
            raise Exception(locale["window_not_found"])

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
    except Exception as exc:
        logging.error("%s: %s", locale["error_getting_window"], exc)
        window = None


class Runner:
    def __init__(self, locale: Translations):
        self.cancelled = True
        self.clicks = 0
        self.init_keybindings()
        self.locale = locale
        self.cold_start = True

    def init_keybindings(self):
        keyboard.on_press(self.handle_keyboard_press)

    def handle_keyboard_press(self, event):
        if event.name == "l":
            self.cancelled = True
            logging.info(self.locale["stopped"])
        elif event.name == "k":
            logging.info(self.locale["running"])
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
                # print(f"{box.cls.item()}; {box.conf.item()}")

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
        logging.info(self.locale["loading"])
        self.model = YOLO("best.pt")
        logging.info(self.locale["ready"])

        while True:
            if self.cancelled:
                time.sleep(0.1)
                continue

            window = get_window(self.locale)

            if not window:
                time.sleep(1)
                continue

            try:
                screenshot = self.grab_screenshot(window)

                # autoreplay feature
                if self.find_replay_button(screenshot, window):
                    continue

                if self.cold_start:
                    logging.info(self.locale["cold_start"])
                    self.cold_start = False

                detected: List[Any] = self.model(screenshot, verbose=False)
                self.detect_figure_and_click(detected, window)
                time.sleep(0.006)
            except Exception as e:
                logging.exception("Error: %s", e)
                continue


def get_user_language():
    while True:
        logging.info(
            "Choose your language (выберите ваш язык):\n\t1 - English, 2 - Русский"
        )

        language_id = int(input())

        if language_id == 1:
            return "en"
        elif language_id == 2:
            return "ru"


if __name__ == "__main__":
    lang: Language = get_user_language()
    locale: Translations = locales[lang]

    runner = Runner(locale)
    runner.run()
