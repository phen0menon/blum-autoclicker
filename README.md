<h1 align="center">Blum Autoclicker (YOLOv8)</h1>

**Features**:

- Extra fast clicker (uses GPU, YOLOv8 pretrained model)
- 100% precision and ignores bombs
- Autoreplay feature (run and go)


**Demo:**

https://github.com/user-attachments/assets/73f00997-9f5f-41e7-8ecf-e2432f6cf3a2


<details>
  <summary>🇺🇸 English instructions</summary>
  <br />

  **Works with the recent (14.07.2024) update.**

  ### Installation

  0. You will need Nvidia GPU to run this app since it uses CUDA cores to achieve fast speed.
  1. Download the repository (https://github.com/phen0menon/blum-autoclicker/releases)
  2. Install Python >= 3.8 (https://www.python.org/downloads/)
  3. Open the cmd or powershell in the project folder (blum-autoclicker).
  4. Install requirements (run in the cmd). Copy and paste the line, not the whole text!:
```
# Base requirements:
      
pip install -r requirements.txt

# PyTorch with CUDA enabled (required!):
      
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
  4. Run the process:
  ```
  python main.py
  ```
  5. Follow instructions given in the cmd

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

0. Понадобится видеокарта от Nvidia (используем CUDA ядра, чтобы эффективно распознавать изображение)
1. Скачайте репозиторий (https://github.com/phen0menon/blum-autoclicker/releases)
2. Нужен Python >= 3.8 (https://www.python.org/downloads/)
3. Откройте cmd или powershell в папке проекта (blum-autoclicker)
4. Установка зависимостей (запустите в командной строке). Нужно скопировать именно строчки команд, не весь текст!:
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
