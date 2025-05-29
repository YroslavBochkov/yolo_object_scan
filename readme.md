# YOLO Object Scan

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)](https://opencv.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLOv8-orange?logo=yolo)](https://github.com/ultralytics/ultralytics)
[![NumPy](https://img.shields.io/badge/NumPy-1.26.x-blueviolet?logo=numpy)](https://numpy.org/)

Легковесное приложение для обнаружения и подсчёта объектов на изображении или видеопотоке с помощью нейросети YOLOv8.

## Возможности

- Детектирует все объекты из датасета COCO (80 классов)
- Работает с видеопотоками (например, с публичных камер или локальных файлов)
- Визуализирует найденные объекты и их количество прямо на изображении
- Выводит статистику по каждому классу объектов в консоль и на экран

## Установка

1. Клонируйте репозиторий и перейдите в папку проекта.
2. Создайте и активируйте виртуальное окружение:
    ```
    python3 -m venv venv
    source [venv/bin/activate](VALID_FILE)
    ```
3. Установите зависимости:
    ```
    pip install -r [requirements.txt](VALID_FILE)
    ```
3. По умолчанию используется тестовый публичный видеопоток. Чтобы использовать свой, замените переменную `stream_url` в файле `detect_people.py` на нужную ссылку (например, на локальный mp4-файл или прямой поток камеры).

## Пример кода

```python
from ultralytics import YOLO
import cv2
from coco_classes import COCO_CLASSES

model = YOLO('yolov8n.pt')
img = cv2.imread('input.jpg')
results = model(img)
for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    if confidence > 0.4:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f'{COCO_CLASSES[class_id]} {confidence:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
cv2.imwrite('output.jpg', img)
```

**Автор:** Ярослав Бочков

## Лицензия

MIT License