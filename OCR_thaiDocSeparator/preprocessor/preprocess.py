import cv2

def resize_keep_aspect_ratio(image, width=None, height=None):
    # Get the dimensions of the image
    h, w = image.shape[:2]

    # If width and height are both None, return the original image
    if width is None and height is None:
        return image

    # If only the width is None, calculate it from the height and aspect ratio
    if width is None:
        ratio = height / float(h)
        width = int(w * ratio)

    # If only the height is None, calculate it from the width and aspect ratio
    if height is None:
        ratio = width / float(w)
        height = int(h * ratio)

    # Resize the image using the calculated dimensions
    resized_image = cv2.resize(image, (width, height))

    return resized_image

def preprocess(IMAGE_PATH):
  image = cv2.imread(IMAGE_PATH)
  image = resize_keep_aspect_ratio(image, width=4096)
  rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Remove paper edge area from scanning
  # rgb_image = rgb_image[50:-50,50:-50]

  # Binarization: Otsu's Binarization
  p_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY) 
  p_image = cv2.threshold(p_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


  return rgb_image, p_image