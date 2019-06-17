try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

m=pytesseract.image_to_string(Image.open('image_bin.jpg'))
print(m)
# def find_keywords():
#     pattern = 

# #https://github.com/UB-Mannheim/tesseract/wiki
# #Link for OCR engine 
