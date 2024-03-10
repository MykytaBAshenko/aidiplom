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


from PIL import Image, ImageDraw

def create_shape_image(shape_coordinates, fill_image_path, output_filename, desired_width, desired_height):
  """
  Creates a PNG image of a specified shape filled with another image,
  with a specified output image size.
  """

  # Open the fill image
  fill_image = Image.open(fill_image_path)

  # Resize the fill image to match the desired output size
  fill_image = fill_image.resize((desired_width, desired_height))

  # Create the output image with the desired size
  output_image = Image.new("RGB", (desired_width, desired_height))

  # Create the mask with the same size as the output image
  mask_image = Image.new("L", output_image.size, 0)
  draw = ImageDraw.Draw(mask_image)

  # Draw the shape onto the mask, using scaled coordinates
  scaled_coordinates = [(x * desired_width / fill_image.width, y * desired_height / fill_image.height) for x, y in shape_coordinates]
  draw.polygon(scaled_coordinates, fill=255)

  # Paste the fill image onto the output image using the mask
  output_image.paste(fill_image, (0, 0), mask=mask_image)

  # Save the output image as PNG
  output_image.save(output_filename, "PNG")

# Example usage: Define heart shape coordinates (replace with your desired shape)
heart_coordinates = [
  (50, 100), (123, 0.1), (500,111), (1000, 234), (0.1, 432)  # Coordinates relative to image size
]

# Example usage: Specify path to your fill image and output filename
fill_image_path = "11.jpg"
output_filename = "heart_image.png"
desired_width = 1500  # Desired width of the output image
desired_height = 500  # Desired height of the output image

create_shape_image(heart_coordinates, fill_image_path, output_filename, desired_width, desired_height)

print(f"Shape image created: {output_filename}")