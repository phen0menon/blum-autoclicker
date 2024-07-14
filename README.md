## Blum Autoclicker (автокликер для BLUM)

**Работает с последним (4 июля 2024) обновлением игры**.

![2024-07-04 19-27-47 (online-video-cutter com) (4)](https://github.com/phen0menon/blum-autoclicker/assets/15520523/dcf4943c-4086-4322-8d42-b1f1e3fd6009)

<details>
  <summary>🇺🇸 English instructions</summary>
  <br />

  **Works with the recent (04.07) recolorization update.**

  ### Installation
  
  1. Clone the repository
  2. Install Python >= 3.8 (https://www.python.org/downloads/)
  3. Install requirements:
  ```
  # base requirements
  pip install -r requirements.txt
  # pytorch with CUDA enabled
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
  3. Run in the cmd:
  ```
  python main.py
  ```
  4. Follow instructions given in the cmd

  ### Possible problems

  All problems may occur because of PyTorch installed without CUDA support. To fix that, run the following commands:
  ```
  # uninstall existing packages first
  pip uninstall torch torchvision torchaudio
  
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
  
</details>

<details>
  <summary>🇷🇺 Russian instructions</summary>

  ### Установка:

1. Клонируйте репозиторий (скачать)
2. Нужен Python >= 3.8 (https://www.python.org/downloads/)
3. Установка зависимостей:
```
# Общие зависимости проекта
pip install -r requirements.txt

# Пакеты, чтобы компьютерное зрение работало на GPU, а не на CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
4. Запустить скрипт:
```
python main.py
```
5. Следовать инструкции :)

### Возможные проблемы

<details>
  <summary>Кликер медленно работает!</summary>

  Нужно переустановить PyTorch с CUDA:
  ```
  # uninstall existing packages first
  pip uninstall torch torchvision torchaudio
  
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
</details>

<details>
  <summary>Вылазит какая-то ошибка!</summary>

  Попробовать переустановить PyTorch с CUDA:

  ```
  # uninstall existing packages first
  pip uninstall torch torchvision torchaudio
  
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

  Если ошибка все еще есть - создавайте [issue](https://github.com/phen0menon/blum-autoclicker/issues) 
</details>
</details>

### Disclaimer

**NOTE:** I created this project for education purposes in the ML field.

### License

© Rolan Ibragimov (phen0menon) 2024. Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) License.
