try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

print(pytesseract.image_to_string(Image.open('text.jpg')))

#https://github.com/UB-Mannheim/tesseract/wiki
#Link for OCE engine 
