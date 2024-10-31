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


def setup_logger(logger):
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    sh.setFormatter(formatter)

    def decorate_emit(fn):
        def new(*args):
            levelno = args[0].levelno
            if levelno >= logging.ERROR:
                color = '\x1b[31;1m'
            elif levelno >= logging.INFO:
                color = "\x1b[32;1m"
            elif levelno >= logging.DEBUG:
                color = "\x1b[35;1m"
            else:
                color = "\x1b[0m"
            args[0].msg = "{0}{1}\x1b[0m".format(color, args[0].msg)
            args[0].args = tuple("\x1b[1m" + str(arg) + "\x1b[0m" for arg in args[0].args)
            return fn(*args)

        return new

    sh.emit = decorate_emit(sh.emit)
    logger.addHandler(sh)


logger = logging.getLogger()
setup_logger(logger=logger)

Language = Literal["en", "ru"]
Translations = Dict[str, str]

locales: Dict[Language, Translations] = {
    "en": {
        "welcome": "Welcome to Blum AutoClicker bot!\nAuthor: https://github.com/phen0menon"
        + "\nDonations are always appreciated:\n- BTC: 1F97kP3UjQXT1oAd14eiBe6gsBuwoXAqZN"
        + "\n- USDT (TRC20): TEG9LE43HFfLAkGC8sas1swXYQkowsLntn\n"
        + "- TON: UQDJuHGnRvs9FPtX6kyUuELpkh2ZbUyS9uB-VIjZqUu9Ih_C"
        + "\n\nBefore starting, make sure that:\n1. You have Telegram Desktop opened"
        + "\n2. You have selected Dark Mode in the Telegram so autoplay feature could work!\n",
        "selected": "Language selected",
        "loading": "Loading pretrained model...",
        "exiting": "Exiting",
        "window_not_found": "Window not found, did you open BLUM app?",
        "error_getting_window": "Error getting window",
        "ready": "Ready: Start the Drop Game and then press 'k' to start, 'l' to stop, CTRL + C to exit",
        "cold_start": "Model is running first time, may take a few seconds...",
        "stopped": "⏸️  Stopped, press 'k' to start again",
        "running": "🔥 Running, press 'l' to stop",
    },
    "ru": {
        "welcome": "Добро пожаловать в бота Blum AutoClicker!\nАвтор: https://github.com/phen0menon"
        + "\nДонаты всегда приветствуются:\n- BTC: 1F97kP3UjQXT1oAd14eiBe6gsBuwoXAqZN"
        + "\n- USDT (TRC20): TEG9LE43HFfLAkGC8sas1swXYQkowsLntn"
        + "\n- TON: UQDJuHGnRvs9FPtX6kyUuELpkh2ZbUyS9uB-VIjZqUu9Ih_C"
        + "\n\nПеред началом убедитесь, что:\n1. У вас открыт Telegram Desktop"
        + "\n2. Вы выбрали темный режим в Telegram, чтобы функция автовоспроизведения работала!\n",
        "selected": "Выбран язык",
        "exiting": "Выходим из бота",
        "loading": "Загрузка обученной модели...",
        "window_not_found": "Окно не найдено, вы открыли приложение BLUM?",
        "error_getting_window": "Ошибка получения окна",
        "ready": "Готово: Запустите мини-игру и нажмите 'k' для начала, 'l' для остановки, CTRL + C для выхода",
        "cold_start": "Модель запускается впервые, это может занять несколько секунд...",
        "stopped": "⏸️  Остановлено, для запуска снова нажмите 'k'",
        "running": "🔥 Запущено, для остановки нажмите 'l'",
    },
}

LANGUAGES_BY_CODES = {"en": "English", "ru": "Русский"}


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
            logging.debug(self.locale["stopped"])
        elif event.name == "k":
            logging.debug(self.locale["running"])
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
        logging.debug(self.locale["loading"])
        self.model = YOLO("best.pt")
        logging.debug(self.locale["ready"])

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
                    logging.debug(self.locale["cold_start"])
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
            "Choose your language (выберите ваш язык):\n1 - English, 2 - Русский\nLanguage:"
        )

        try:
            language_id = int(input())

            if language_id == 1:
                return "en"
            elif language_id == 2:
                return "ru"
        except:
            pass


if __name__ == "__main__":
    try:
        lang: Language = get_user_language()
        locale: Translations = locales[lang]
        logging.debug(f"{locale['selected']}: {LANGUAGES_BY_CODES[lang]}\n")
        logging.info(locale["welcome"])

        runner = Runner(locale)
        runner.run()
    except KeyboardInterrupt:
        logging.debug(f"{locale['exiting']}...")
