1. Install conda
2. Activate environment and load packages
```
conda create -n ocr python=3.8
conda activate ocr
conda install pip
pip install -r requirements.txt
pip install -r OCR_thaiDocSeparator\requirements.txt
```
3. Run script 
```
uvicorn api:app --host 0.0.0.0 --port 5000
```
4. API documents & Test: go to localhost:5000/docs