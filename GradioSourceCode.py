import cv2
import numpy as np
import io
import requests
from PIL import Image
import gradio as gr


# Function for sky pixel identification
def identify_sky(original_image):
  # Convert the image to NumPy array in RGB format
  image = cv2.cvtColor(np.array(original_image), cv2.COLOR_BGR2RGB)

  # Convert the image to HSV color space
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  # Define lower and upper bounds for blue sky color in HSV
  lower_blue = np.array([100, 50, 50])
  upper_blue = np.array([130, 255, 255])

  # Threshold the image to identify sky pixels
  mask = cv2.inRange(hsv, lower_blue, upper_blue)

  # Apply morphological operations to clean up the mask
  kernel = np.ones((5, 5), np.uint8)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

  # Apply the mask to the original image
  sky = cv2.bitwise_and(image, image, mask=mask)

  return image, mask, sky


def resize_to_match(image, target_image_url):
  # Load Van Gogh's starry night image
  starry_night_response = requests.get(target_image_url)
  starry_night_image = Image.open(io.BytesIO(starry_night_response.content))

  # Convert the image to a NumPy array in RGB format
  starry_night_image_cv2 = cv2.cvtColor(np.array(starry_night_image), cv2.COLOR_BGR2RGB)

  # Calculate the aspect ratio of the starry night image
  starry_night_aspect_ratio = starry_night_image_cv2.shape[1] / starry_night_image_cv2.shape[0]

  # Determine the target dimensions based on image aspect ratios
  # Resize the starry night image to match the target dimensions
  if image.shape[0]/starry_night_image_cv2.shape[0] > image.shape[1]/starry_night_image_cv2.shape[1]:
    target_height = image.shape[0]
    target_width = int(target_height * starry_night_aspect_ratio)
    resized_starry_night = cv2.resize(starry_night_image_cv2, (target_width, target_height))
  else:
    target_width = image.shape[1]
    target_height = int(target_width / starry_night_aspect_ratio)
    resized_starry_night = cv2.resize(starry_night_image_cv2, (target_width, target_height))

  return resized_starry_night


def replace_sky_with_starry_night(image, mask, resized_starry_night):
  # Create a copy of the original image to avoid modifying the input directly
  result = image.copy()

  # Replace the sky pixels with the corresponding pixels from the resized starry night image
  sky_coords = np.where(mask > 0)

  # Replace the sky pixels in the original image with the corresponding pixels from the resized starry night image
  result[sky_coords[0], sky_coords[1]] = resized_starry_night[sky_coords[0], sky_coords[1]]

  return result


def process_image(original_image):
  # Identify sky pixels in the original image
  image, mask, sky = identify_sky(original_image)

  # Resize Van Gogh's starry night image to match the dimensions of the original image
  resized_starry_night = resize_to_match(image, "https://travelmate.tech/media/images/cache/new_york_museum_of_modern_art_03_notte_stellata_van_gogh_jpg_1920_1080_cover_70.jpg")

  # Replace the sky in the original image with Van Gogh's starry night
  result = replace_sky_with_starry_night(image, mask, resized_starry_night)

  # Convert the final result to RGB color space for Gradio display
  final_image = cv2.cvtColor(np.array(result), cv2.COLOR_BGR2RGB)

  return final_image


# Gradio Interface
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(),
    outputs=gr.Image(type="pil"),
    title="Sky Pixel Identification",
    description="Replace the sky pixel with starry night."
)


iface.launch(share=True, debug=True)
