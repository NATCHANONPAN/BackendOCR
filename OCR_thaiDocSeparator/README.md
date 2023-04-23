# Thai Document Words Separator

## Pipeline
![pipeline](https://github.com/Gyoowai/OCR_thaiDocSeperator_2022/blob/master/pictures/pipeline.png)

## Algorithm
1. Preprocess
    - Remove paper edge area from scanning
    - Binarization using Otsu's Binarization
2. Row Separator
    - Thinning: morphological opening
    - Extract row histogram sequence from image
    - Sequence smoothing 
    - Peak detection using [avhn's Peakdetect](https://github.com/avhn/peakdetect)
    - Calculate the boundary line: every peaks are the center of the row and closest two valleys are the boundary of that row.

3. Word Separator
    - Morphological Dilation
    - Extract core text area of image
    - Extract column histogram sequence from image
    - Find adhering space area in sequence (space area is the column that has no character in it)
    -  Calculate the boundary line


## Usage
```
from OCR_thaiDocSeparator_2022.Detector import detector
detector(IMG_PATH, OUT_DIR_PATH)
```

## Example
[Example Notebook](example.ipynb)
