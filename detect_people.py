from ultralytics import YOLO
import cv2
from coco_classes import COCO_CLASSES
from collections import Counter

# Используйте более точную модель, если позволяет производительность:
# model = YOLO('yolov8s.pt')  # small
# model = YOLO('yolov8m.pt')  # medium
model = YOLO('yolov8n.pt')  # nano (самая быстрая, но менее точная)


def get_color(class_id):
    """Генерирует уникальный цвет для каждого класса."""
    import random
    random.seed(class_id)
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )


def analyze_video_stream(stream_url):
    '''
    Анализирует видеопоток, определяет и подписывает все объекты на каждом
    кадре, выводит результат на экран.
    '''
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f'Не удалось открыть видеопоток: {stream_url}')
        return

    print('Нажмите "q" для выхода.')
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Поток завершён или ошибка чтения кадра.')
            break

        # (Опционально) Увеличьте разрешение кадра для лучшего распознавания
        # frame = cv2.resize(frame, (1280, 720))

        # Детекция объектов
        results = model(frame)

        # Увеличьте порог уверенности для уменьшения ложных срабатываний
        CONFIDENCE_THRESHOLD = 0.5  # 0.5-0.6 для большей точности

        class_counts = Counter()
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if confidence >= CONFIDENCE_THRESHOLD:
                class_counts[class_id] += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = get_color(class_id)
                label = f'{COCO_CLASSES[class_id]} {confidence:.2f}'
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), color, 2
                )
                cv2.putText(
                    frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
                )

        # Сформировать строку с количеством объектов каждого класса
        counts_str = ', '.join(
            f'{COCO_CLASSES[cid]}: {count}'
            for cid, count in class_counts.items()
        )
        # Вывести эту строку на изображение
        cv2.putText(
            frame,
            counts_str,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        # Показать изображение в окне
        cv2.imshow('YOLO Object Detection', frame)

        # Показать количество объектов каждого класса в консоли
        print(f'Объекты на кадре: {counts_str}', end='\r')

        # Выход по клавише 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Для видеопотока укажите ссылку на камеру:
    stream_url = (
        'https://webcam.vliegveldzeeland.nl:7171/axis-cgi/mjpg/video.cgi'
    )
    analyze_video_stream(stream_url)
