import cv2
from .preprocessor.preprocess import preprocess
from .rowSeparator.rowSeparator import rowSeparator
from .wordSeparator.wordSeparator import wordSeparator

def detector(IMG_PATH, OUT_DIR_PATH):
    rgb_image, p_image = preprocess(IMG_PATH)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    box_image = rgb_image.copy()
    lines = rowSeparator(p_image)

    res = []
    for y_upper, y_lower in lines: 
        imgs = wordSeparator(rgb_image[y_upper:y_lower,:], p_image[y_upper:y_lower,:])
        for x_left, x_right in imgs: 
            start_point = (x_left, y_upper)
            end_point = (x_right, y_lower)
            color = (0,255,0)
            thickness = 10
            cv2.rectangle(box_image, start_point, end_point, color, thickness)
            res.append(rgb_image[y_upper:y_lower, x_left:x_right])

    cv2.imwrite(OUT_DIR_PATH, box_image)

    return res