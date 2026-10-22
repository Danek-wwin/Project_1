import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np

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
    # Мы принимаем объект Image от Streamlit, а не путь к файлу
    img = image.convert('L')
    img = ImageOps.invert(img)
    arr = np.array(img)

    threshold = arr.mean() + 30
    arr = np.where(arr > threshold, 255, 0).astype(np.uint8)

    # Ищем границы по столбцам
    cols = np.any(arr > 0, axis=0)
    col_indices = np.where(cols)[0]

    if len(col_indices) == 0:
        return []

    # Разбиваем на группы
    splits = np.split(col_indices, np.where(np.diff(col_indices) > 10)[0] + 1)

    tensors = []
    for split in splits:
        if len(split) < 5: # Игнорируем мелкий мусор
            continue

        x_min, x_max = split[0], split[-1]
        rows = np.any(arr[:, x_min:x_max+1] > 0, axis=1)
        y_min, y_max = np.where(rows)[0][[0, -1]]

        digit_arr = arr[y_min:y_max+1, x_min:x_max+1]

        h, w = digit_arr.shape
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

# Кнопка загрузки файла
uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Показываем загруженное фото
    st.image(image, caption='Ваше фото', width=300)
    
    with st.spinner('Распознаем...'):
        tensors_list = preprocess_multiple_digits(image)
        
        if len(tensors_list) == 0:
            st.error("Не удалось найти цифры на фото. Попробуйте сфотографировать еще раз.")
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