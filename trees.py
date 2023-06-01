import os
import tensorflow as tf

# Путь для сохранения модели
model_path = "tree_model.h5"

# Загрузка данных изображений для обучения
train_image_dir = "images"
train_image_files = [f for f in os.listdir(train_image_dir) if os.path.isfile(os.path.join(train_image_dir, f))]
check_image_dir = "images_check"
check_image_files = [f for f in os.listdir(check_image_dir) if os.path.isfile(os.path.join(check_image_dir, f))]

# Проверка наличия изображений для обучения
if len(train_image_files) == 0:
    print("Отсутствуют изображения для обучения в папке 'images'")
    exit()

# Создание списка меток для обучающих данных (дерево - 1, не дерево - 0)
train_labels = []
for file in train_image_files:
    if "tree" in file:
        train_labels.append(1)
    else:
        train_labels.append(0)

# Загрузка и предварительная обработка изображений для обучения
train_image_data = []
for file in train_image_files:
    image_path = os.path.join(train_image_dir, file)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    train_image_data.append(image)

# Преобразование данных в формат TensorFlow
train_image_data = tf.stack(train_image_data)
train_labels = tf.convert_to_tensor(train_labels)

# Создание модели нейронной сети
model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
model.trainable = False

# Добавление слоев классификации поверх предобученной модели
flatten_layer = tf.keras.layers.Flatten()(model.output)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(flatten_layer)
model = tf.keras.models.Model(inputs=model.input, outputs=output_layer)

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(train_image_data, train_labels, epochs=10, verbose=1)

# Сохранение модели
model.save(model_path)
print(f"Модель сохранена по пути: {model_path}")

# Запрос пользователю на поиск деревьев в изображениях
user_input = input("Вы хотите выполнить поиск деревьев в изображениях? (Yes/No): ")
if user_input.lower() == "yes":
    # Загрузка сохраненной модели
    model = tf.keras.models.load_model(model_path)
    print("Загружена сохраненная модель")

    # Поиск деревьев в изображениях в текущей папке
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(current_dir, "images")
    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]

    # Проверка наличия изображений
    if len(image_files) == 0:
        print("Отсутствуют изображения в папке 'images'")
        exit()

    # Загрузка и предварительная обработка изображений для поиска
    search_image_data = []
    for file in image_files:
        image_path = os.path.join(images_dir, file)
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        search_image_data.append(image)

    # Преобразование данных в формат TensorFlow
    search_image_data = tf.stack(search_image_data)

    # Проверка изображений и вывод результатов
    for file, image in zip(image_files, search_image_data):
        image = tf.expand_dims(image, axis=0)
        prediction = model.predict(image)[0][0]
        if prediction > 0.5:
            print(f"Обнаружено дерево на изображении: {file}")
        else:
            print(f"No tree in {file}")
else:
    print("Программа завершена.")
