import cv2
import pytesseract

# Carregando a imagem do documento
imagem_documento_path = 'testpdf.jpg'
imagem_documento = cv2.imread(imagem_documento_path)

# Convertendo a imagem para escala de cinza
imagem_documento_cinza = cv2.cvtColor(imagem_documento, cv2.COLOR_BGR2GRAY)

# Aplicando binarização (thresholding)
_, imagem_documento_binaria = cv2.threshold(imagem_documento_cinza, 128, 255, cv2.THRESH_BINARY)

# Substitua pelo caminho correto no seu sistema
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Detecção de contornos na imagem
contornos, _ = cv2.findContours(imagem_documento_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterando sobre os contornos encontrados
for contorno in contornos:
    # Ignorando contornos muito pequenos
    if cv2.contourArea(contorno) > 100:
        # Obtendo as coordenadas do retângulo que envolve o contorno
        x, y, w, h = cv2.boundingRect(contorno)
        
        # Desenhando o retângulo na imagem original
        cv2.rectangle(imagem_documento, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Recortando a região de texto da imagem original
        regiao_texto = imagem_documento[y:y + h, x:x + w]
        
        # Aplicando OCR na região de texto
        texto_detectado = pytesseract.image_to_string(regiao_texto, lang='por')  # Use o idioma adequado
        
        # Imprimindo o texto detectado
        print(f'Texto no documento: {texto_detectado}')

# Exibindo a imagem com os contornos e retângulos
cv2.imshow('Imagem com Contornos', imagem_documento
)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Imagem Processada', imagem_documento_binaria)
cv2.waitKey(0)
cv2.destroyAllWindows()