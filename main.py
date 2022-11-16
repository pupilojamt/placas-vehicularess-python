# IMPORTACION DE LIRERIAS
import cv2
import numpy as np
import pytesseract
from PIL import Image

# REALIZAMOS LA VIDEO CAPTURA
cap = cv2.VideoCapture(0)
text = ''

# REALIZAMOS EL BUVLE PARA LA DETECCION
while True:
    # REALIZAMOS LA LECTURA DE LA VIDEO CAPTURA
    ret, frame = cap.read()

    if ret == False:
        break

    # DIBUJAMOS UN RECTANGULO
    cv2.rectangle(frame, (870, 750), (1070, 850), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, text[0:7], (900, 810), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # EXTRAEMOS EL ANCHO Y ALTO DE LO FOTOGRAMAS
    al, an, c = frame.shape

    # TOMAR EL CENTRO DE LA IMAGEN
    # en x
    x1 = int(an / 3)    #tomas el 1/3 de la imagen
    x2 = int(x1 * 2)    #hasta el inicio del 3/3 de la imagen

    # en y
    y1 = int(al / 3)    #tomas el 1/3 de la imagen
    y2 = int(y1 * 2)    #hasta el inicio del 3/3 de la imagen

    # TEXTO
    cv2.rectangle(frame, (x1 + 160, y1 + 500), (1120, 940), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, 'Procesando Placa', (x1 + 180, y1 + 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # UBICAMOS EL RECTANGULO EN LAS ZONAS EXTRAIDAS
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # REALIZAMOS EL RECORTE A NUESTRA ZONA DE INTERES
    recorte = frame[y1:y2, x1:x2]

    # PROCESAMIENTO DE LA ZONA DE INTERES
    mB = np.matrix(recorte[:, :, 0])
    mG = np.matrix(recorte[:, :, 1])
    mR = np.matrix(recorte[:, :, 2])

    # COLOR
    Color = cv2.absdiff(mB, 255)

    # BINARIZAAMOS LA IMAGEN
    _, umbral = cv2.threshold(Color, 45, 255, cv2.THRESH_BINARY)

    # EXTRAEMOS LOS CONTORNOS DE LA ZONA SELECCIONADA
    contornos, _ = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # PRIMERO ORDENAMOS DEL MAS GRANDE AL MAS PEQUENO
    contornos = sorted(contornos, key=lambda x: cv2.contourArea(x), reverse=True)

    # DIBUJAMOS LOS CONTORNOS EXTRAIDOS
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 500 and area < 5000:
            # DETECTAMOS LA PLACA
            x, y, ancho, alto = cv2.boundingRect(contorno)

            # EXTRAEMOS LA COORDENADA
            xpi = x + x1
            ypi = y + y1

            xpf = x + ancho + x1
            ypf = y + alto + y1

            # DIUJAMOS EL RECTANGULO
            cv2.rectangle(frame, (xpi, ypi), (xpf, ypf), (255, 255, 0), 2)

            # EXTRAEMOS LO PIXELES
            placa = frame[ypi:ypf, xpi:xpf]

            # EXTRAEMOS EL ANCHO Y ALTO DE LOS FOTOGRAMAS
            alp, anp, cp = placa.shape

            # PROCESAMOS LOS PIXELES PARA EXTRAER LOS VALORES DE LAS PLACAS
            Mva = np.zeros((alp,anp))

            # NORMALIZAMOS LAS MATRICES
            mBp = np.matrix(placa[:, :, 0])
            mGp = np.matrix(placa[:, :, 1])
            mRp = np.matrix(placa[:, :, 2])

            # CREAMOS UNA MASCARA
            for col in range(0, alp):
                for fil in range(0, anp):
                    Max = max(mRp[col, fil], mGp[col, fil], mBp[col, fil])
                    Mva[col, fil] = 255 - Max

            # BINARIZAMOS LA IMAGEN
            _, bin = cv2.threshold(Mva, 150, 255, cv2.THRESH_BINARY)

            # CONVERTIMOS LA MATRIZ DE LA IMAGEN
            bin = bin.reshape(alp, anp)
            bin = Image.fromarray(bin)
            bin = bin.convert("L")

            # NOS ASEGURAMOS DE TENER UN BUEN TAMANO DE PLACA
            if alp >= 36 and anp >= 82:
                # DECLARAMOSS LA DIRECCION DE PYTESSERACT
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

                # EXTRAEMOS EL TEXTO
                config = "--psm 1"
                texto = pytesseract.image_to_string(bin, config=config)

                # IF PARA NO MOSTRAR BASURA
                if len(texto) >= 7:
                    text = texto
                    print(text)

            break

    # MOSTRAMOS EL RECORTE EN GRIS
    cv2.imshow("Vehiculos", frame)

    # LEEMOS UNA TECLA
    t = cv2.waitKey(1)

    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()