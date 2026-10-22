import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import cv2
from streamlit_cropper import st_cropper

# ===== 1. АРХИТЕКТУРА (скопирована из твоего кода) =====
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# ===== 2. ЗАГРУЗКА МОДЕЛИ =====
# @st.cache_resource нужен, чтобы модель загружалась в память только один раз
@st.cache_resource
def load_model():
    model = Net()
    # Загружаем веса, которые лежат в той же папке, что и app.py
    model.load_state_dict(torch.load('mnist_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# ===== 3. ПРЕПРОЦЕССИНГ (скопирован из твоего кода) =====
def preprocess_multiple_digits(image):
    img = image.convert('L')
    img = ImageOps.invert(img)
    arr = np.array(img)

    # 1. Бинаризация методом Оцу (отлично работает при разном освещении)
    _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Убираем мелкий шум (текст внизу, точки)
    # Мы "размываем" картинку, чтобы мелкие элементы исчезли, а крупные остались
    kernel = np.ones((2,3), np.uint8)
    arr = cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)

    # 3. Находим контуры всех объектов на фото
    contours, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tensors = []
    
    # 4. Оставляем только достаточно большие контуры (игнорируем мелкий текст)
    # Высота картинки может быть разной, поэтому берем порог относительно размера
    min_height = arr.shape[0] * 0.05 # Игнорируем всё, что меньше 5% высоты фото

    # Сортируем контуры слева направо, чтобы цифры шли по порядку
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes.sort(key=lambda x: x[0])

    for x, y, w, h in bounding_boxes:
        if h < min_height: # Пропускаем логотипы и мелкий текст
            continue
            
        # 2. НОВОЕ: Отсекаем слишком вытянутые объекты (тени, линии, края стола)
        # Если объект в 4 раза шире, чем выше (или наоборот) — это не цифра, а мусор.
        aspect_ratio = w / float(h)
        if aspect_ratio > 4.0 or aspect_ratio < 0.25:
            continue

        # 3. НОВОЕ: Проверяем "заполненность" (плотность белых пикселей)
        # У цифры линия занимает около 20-40% от рамки. Если заполнено 90% — это сплошное пятно тени.
        roi = arr[y:y+h, x:x+w]
        white_pixels = np.sum(roi == 255)
        total_pixels = w * h
        fill_ratio = white_pixels / total_pixels
    
        if fill_ratio > 0.7: # Если пятно слишком плотное (залитое) — пропускаем
            continue

        # Вырезаем найденную цифру
        digit_arr = arr[y:y+h, x:x+w]

        # Вписываем в квадрат 28x28
        size = max(h, w)
        pad = int(size * 0.2)
        square = np.zeros((size + 2*pad, size + 2*pad), dtype=np.uint8)
        y_off = (size + 2*pad - h) // 2
        x_off = (size + 2*pad - w) // 2
        square[y_off:y_off+h, x_off:x_off+w] = digit_arr

        img_digit = Image.fromarray(square).resize((28, 28), Image.LANCZOS)
        tensor = transforms.ToTensor()(img_digit)
        tensor = transforms.Normalize((0.1307,), (0.3081,))(tensor)

        tensors.append(tensor)

    return tensors

# ===== 4. ИНТЕРФЕЙС STREAMLIT =====
st.set_page_config(page_title="Распознавание чисел", page_icon="🔢")

st.title("🔢 Распознавание рукописных чисел")
st.write("Загрузите фото с числом. Цифры должны быть написаны **темным маркером на светлом фоне** с небольшим пробелом друг от друга.")

# --- Выбор способа ввода: камера или файл ---
st.write("Выберите способ добавления фото:")
option = st.radio("", ["📷 Сделать фото", "📁 Загрузить файл"], label_visibility="collapsed")

image = None

if option == "📷 Сделать фото":
    picture = st.camera_input("Сделайте снимок")
    if picture:
        image = Image.open(picture)
else:
    uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

if image is not None:
    st.write("### Обрежьте фото")
    st.write("Выделите рамкой только те цифры, которые нужно распознать.")
    
    # Интерактивная обрезка
    cropped_image = st_cropper(
        image, 
        realtime_update=True, 
        box_color='#FF0000', 
        aspect_ratio=None
    )
    
    st.image(cropped_image, caption='Фото после обрезки', width=300)
    
    with st.spinner('Распознаем...'):
        # Передаем в функцию ОБРЕЗАННОЕ фото
        tensors_list = preprocess_multiple_digits(cropped_image)
        
        if len(tensors_list) == 0:
            st.error("Не удалось найти цифры на фото. Попробуйте выделить их рамкой.")
        else:
            predicted_number = ""
            confidences = []
            
            with torch.no_grad():
                for t in tensors_list:
                    t = t.unsqueeze(0)
                    output = model(t)
                    probs = F.softmax(output, dim=1)[0]
                    pred = probs.argmax().item()
                    conf = probs[pred].item() * 100
                    predicted_number += str(pred)
                    confidences.append(conf)
            
            # Выводим большой красивый результат
            st.success(f"### Распознанное число: {predicted_number}")
            
            # Выводим уверенность по каждой цифре
            st.write("Детали по цифрам:")
            for i, (digit, conf) in enumerate(zip(predicted_number, confidences)):
                st.write(f"- Цифра {i+1}: **{digit}** (уверенность {conf:.1f}%)")
                
            # --- БЛОК С КНОПКОЙ ---
            st.divider()

            if 'show_images' not in st.session_state:
                st.session_state.show_images = False

            button_text = "🙈 Скрыть, как видит нейросеть" if st.session_state.show_images else "👁️ Показать, как видит нейросеть"

            if st.button(button_text):
                st.session_state.show_images = not st.session_state.show_images
                st.rerun()

            if st.session_state.show_images:
                st.write("Ниже показаны картинки 28x28, которые нейросеть получила на вход:")
    
                cols = st.columns(len(tensors_list))
    
                for i, tensor in enumerate(tensors_list):
                    img_array = tensor[0].numpy()
                    img_array = img_array * 0.3081 + 0.1307
                    
                    # Обрати внимание: в конце дописано 'clamp=True'
                    cols[i].image(img_array, caption=f"Вход для {predicted_number[i]}", width=100, clamp=True)
