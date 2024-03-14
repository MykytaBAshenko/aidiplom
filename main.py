# from colorthief import ColorThief
# import matplotlib.pyplot as plt
# import colorsys
# ct = ColorThief("22.jpg")
# dominat_color = ct.get_color(quality=1)

# plt.imshow([[dominat_color]])

# palette = ct.get_palette(color_count = 5)
# plt.imshow([[palette[i] for i in range(5)]])
# plt.show()

# for color in palette: 
#     print(color)
#     print(f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}")
#     print(colorsys.rgb_to_hsv(*color))
#     print(colorsys.rgb_to_hls(*color))
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
# import cv2
# import numpy as np


# def create_bar(height, width, color):
#     bar = np.zeros((height, width, 3), np.uint8)
#     bar[:] = color
#     red, green, blue = int(color[2]), int(color[1]), int(color[0])
#     return bar, (red, green, blue)

# img = cv2.imread('11.jpg')
# height, width, _ = np.shape(img)
# # print(height, width)

# data = np.reshape(img, (height * width, 3))
# data = np.float32(data)

# number_clusters = 15
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# flags = cv2.KMEANS_RANDOM_CENTERS
# compactness, labels, centers = cv2.kmeans(data, number_clusters, None, criteria, 10, flags)
# # print(centers)

# font = cv2.FONT_HERSHEY_SIMPLEX
# bars = []
# rgb_values = []

# for index, row in enumerate(centers):
#     bar, rgb = create_bar(200, 200, row)
#     bars.append(bar)
#     rgb_values.append(rgb)

# img_bar = np.hstack(bars)

# for index, row in enumerate(rgb_values):
#     image = cv2.putText(img_bar, f'{index + 1}. RGB: {row}', (5 + 200 * index, 200 - 10),
#                         font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
#     print(f'{index + 1}. RGB{row}')

# cv2.imshow('Image', img)
# cv2.imshow('Dominant colors', img_bar)
# # cv2.imwrite('output/bar.jpg', img_bar)

# cv2.waitKey(0)


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# def draw_star(points, base_point, size, color, background_image):
#   """
#   Рисует звезду на холсте с наложением фонового изображения.

#   Args:
#     points: список координат вершин звезды.
#     base_point: координата основания звезды.
#     size: размер изображения.
#     color: цвет звезды.
#     background_image: изображение, используемое в качестве фона.
#   """

#   canvas = np.zeros((size, size, 3), dtype=np.uint8)

#   points = np.array(points) - base_point + size // 2

#   for i in range(len(points)):
#     j = (i + 1) % len(points)
#     plt.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], color=color, linewidth=2)

#   plt.fill(points[:, 0], points[:, 1], color=color)
#   plt.axis("off")

#   background_array = np.array(background_image)

#   final_image = np.copy(background_array)
#   alpha = 0.5
#   final_image[canvas != 0] = (
#       final_image[canvas != 0] * (1 - alpha)
#       + canvas[canvas != 0] * alpha
#   )

#   Image.fromarray(final_image).save("final.png")
#   plt.clf()


# points = [
#   (0, 0),
#   (1, 0.5),
#   (0.87, 0.93),
#   (0.5, 1),
#   (0.13, 0.93),
#   (0, 0.5),
#   (-0.13, 0.93),
#   (-0.5, 1),
#   (-0.87, 0.93),
#   (-1, 0.5),
# ]

# base_point = (0, 0)

# size = 500

# color = "red"

# background_image_path = "22.jpg"

# background_image = Image.open(background_image_path)

# background_image = background_image.resize((size, size))

# draw_star(points, base_point, size, color, background_image)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! RABOTA
# from PIL import Image, ImageDraw

# def create_shape_image(shape_coordinates, fill_image_path, output_filename, desired_width, desired_height):
#   """
#   Creates a PNG image of a specified shape filled with another image,
#   with a specified output image size.
#   """

#   # Open the fill image
#   fill_image = Image.open(fill_image_path)

#   # Resize the fill image to match the desired output size
#   fill_image = fill_image.resize((desired_width, desired_height))

#   # Create the output image with the desired size
#   output_image = Image.new("RGB", (desired_width, desired_height))

#   # Create the mask with the same size as the output image
#   mask_image = Image.new("L", output_image.size, 0)
#   draw = ImageDraw.Draw(mask_image)

#   # Draw the shape onto the mask, using scaled coordinates
#   scaled_coordinates = [(x * desired_width / fill_image.width, y * desired_height / fill_image.height) for x, y in shape_coordinates]
#   draw.polygon(scaled_coordinates, fill=255)

#   # Paste the fill image onto the output image using the mask
#   output_image.paste(fill_image, (0, 0), mask=mask_image)

#   # Save the output image as PNG
#   output_image.save(output_filename, "PNG")

# # Example usage: Define heart shape coordinates (replace with your desired shape)
# heart_coordinates = [
#   (50, 100), (123, 0.1), (0.9, 0.5), (343, 234), (0.1, 432)  # Coordinates relative to image size
# ]

# # Example usage: Specify path to your fill image and output filename
# fill_image_path = "22.jpg"
# output_filename = "heart_image.png"
# desired_width = 500  # Desired width of the output image
# desired_height = 500  # Desired height of the output image

# create_shape_image(heart_coordinates, fill_image_path, output_filename, desired_width, desired_height)

# print(f"Shape image created: {output_filename}")
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! RABOTA


# from PIL import Image, ImageDraw

# def create_shape_image(shape_coordinates, fill_image_path, output_filename, desired_width, desired_height):
#   """
#   Creates a PNG image of a specified shape filled with another image,
#   with a specified output image size.
#   """

#   # Open the fill image
#   fill_image = Image.open(fill_image_path)

#   # Resize the fill image to match the desired output size
#   fill_image = fill_image.resize((desired_width, desired_height))

#   # Create the output image with the desired size
#   output_image = Image.new("RGB", (desired_width, desired_height))

#   # Create the mask with the same size as the output image
#   mask_image = Image.new("L", output_image.size, 0)
#   draw = ImageDraw.Draw(mask_image)

#   # Draw the shape onto the mask, using scaled coordinates
#   scaled_coordinates = [(x * desired_width / fill_image.width, y * desired_height / fill_image.height) for x, y in shape_coordinates]
#   draw.polygon(scaled_coordinates, fill=255)

#   # Paste the fill image onto the output image using the mask
#   output_image.paste(fill_image, (0, 0), mask=mask_image)

#   # Save the output image as PNG
#   output_image.save(output_filename, "PNG")

# # Example usage: Define heart shape coordinates (replace with your desired shape)
# heart_coordinates = [
#   (50, 100), (123, 0.1), (500,111), (1000, 234), (0.1, 432)  # Coordinates relative to image size
# ]

# # Example usage: Specify path to your fill image and output filename
# fill_image_path = "11.jpg"
# output_filename = "heart_image.png"
# desired_width = 1500  # Desired width of the output image
# desired_height = 500  # Desired height of the output image

# create_shape_image(heart_coordinates, fill_image_path, output_filename, desired_width, desired_height)

# print(f"Shape image created: {output_filename}")


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from PIL import Image

# # Define generator network
# class Generator(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(Generator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, output_size),
#             nn.Sigmoid()  # Ensure values are between 0 and 1
#         )

#     def forward(self, x):
#         return self.model(x)

# # Function to generate random noise
# def generate_noise(batch_size, input_size):
#     return torch.rand(batch_size, input_size)

# # Function to generate image from generator output
# def generate_image(generator, noise):
#     img = generator(noise)
#     img = img.view(200, 500, 3)  # Reshape to desired image size and channels
#     img = (img * 255).byte()  # Scale values to 0-255 and convert to bytes
#     return img.numpy()

# # Define colors in RGB format
# colors = [
#     [0, 0, 0],   # Black
#     [255, 255, 255],  # White
#     [128, 128, 128],  # Gray
#     [0, 0, 255],  # Blue
#     [0, 255, 0],  # Green
#     [255, 0, 0],  # Red
#     [255, 255, 0],  # Yellow
#     [128, 0, 128],  # Purple
#     [255, 165, 0],  # Orange
#     [0, 255, 255]  # Cyan
# ]

# # Convert colors to tensors
# colors_tensor = torch.tensor(colors, dtype=torch.float32) / 255

# # Define parameters
# input_size = 100  # Size of input noise vector
# output_size = 500 * 200 * 3  # Size of output image
# batch_size = 1

# # Initialize generator
# generator = Generator(input_size, output_size)

# # Generate noise
# noise = generate_noise(batch_size, input_size)

# # Generate image
# img = generate_image(generator, noise)

# # Convert image to PIL image
# img = Image.fromarray(img)

# # Save image
# img.save('generated_image.png')

# import tensorflow as tf
# import numpy as np

# # Палитра цветов
# palette = [(0, 0, 0), (255, 255, 255), (128, 128, 128), (0, 0, 128), (128, 0, 0)]

# # Размер изображения
# width = 100
# height = 200

# # Модель генератора
# generator = tf.keras.Sequential([
#   tf.keras.layers.Input(shape=(width, height, 1)),
#   tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
#   tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
#   tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
#   tf.keras.layers.Conv2D(5, (3, 3), activation="softmax"),
# ])

# # Функция генерации изображения
# def generate_image(palette):
#   # Случайный шум
#   noise = tf.random.normal((width, height, 1))

#   # Генерация изображения
#   image = generator(noise)

#   # Преобразование к RGB
#   image = tf.argmax(image, axis=-1)
#   image = tf.expand_dims(image, axis=-1)
#   image = tf.cast(image, tf.uint8)
#   image = tf.reshape(image, (width, height, 3))

#   # Палитра
#   image = tf.gather(palette, image)

#   return image

# # Генерация изображения
# image = generate_image(palette)

# # Сохранение изображения
# tf.keras.utils.save_img("image.png", image)


# import cv2
# import numpy as np

# image_size = (512, 512)

# images = []
# for filename in ["1.png", "2.png", "3.png", "4.png", "5.png"]:
#   img = cv2.imread(filename)
#   img = cv2.resize(img, image_size)
#   images.append(img)

# # Replace these with your actual labels (e.g., 0 for digital, 1 for urban)
# labels = [2, 1, 0, 3, 2]

# np.save('train_data.npy', np.array(images))
# np.save('train_labels.npy', np.array(labels))

# print("train_data.npy and train_labels.npy created!")




















# import numpy as np
# import tensorflow as tf
# from PIL import Image

# # Определите палитру цветов
# palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128), (0, 0, 0)]

# # Загрузите обучающие изображения
# images = []
# for i in range(5):
#   image = Image.open(f"{i+1}.png")
#   image = image.resize((512, 512))
#   images.append(np.array(image))

# # Разделите обучающие изображения на массивы входных и выходных данных
# X_train = np.array(images)
# y_train = np.array(images)

# # Создайте модель нейронной сети
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(512, 512, 3)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(512 * 512 * 3, activation='softmax')
# ])

# # Компилируйте модель
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Обучите модель
# model.fit(X_train, y_train, epochs=10)

# # Сгенерируйте новый милитари-рисунок
# new_image = np.zeros((512, 512, 3))
# for i in range(512):
#   for j in range(512):
#     # Сделайте предсказание для каждого пикселя
#     prediction = model.predict(np.array([X_train[0][i][j]]))
#     # Выберите цвет из палитры
#     new_image[i][j] = palette[np.argmax(prediction)]

# # Сохраните изображение
# Image.fromarray(new_image.astype(np.uint8)).save("new_image.png")
import os
import io
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
API_KEY = os.getenv('GOOGLE_API_KEY')
from genai import Client
from PIL import Image

# Function to handle image generation using Gemini API
def generate_image(prompt, max_iterations=100, temperature=0.7):
    client = Client(api_key=API_KEY)  # Get API key (from environment or directly)

    # Prepare prompt
    prompt = prompt.encode('utf-8')

    # Generate image
    response = client.generate_content(
        model_name="art_imagination_prompt",  # Adjust model name as needed
        prompt=prompt,
        max_iterations=max_iterations,
        temperature=temperature,
    )

    # Decode and convert image to PIL format
    image_bytes = response.content.decode('base64')
    image = Image.open(io.BytesIO(image_bytes))
    return image

# Example usage (replace with your desired user input/functionality)
prompt = input("Enter your image prompt: ")
generated_image = generate_image(prompt)

# Optional: Display or save the generated image
generated_image.show()  # Display the image in a window
# generated_image.save("genera