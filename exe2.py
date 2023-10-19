import cv2
import numpy as np
import pygame

#Inciar a webcam
cap = cv2.VideoCapture(1)

# Configura o detector de movimento
#cria um objeto de subtração de plano de fundo do tipo MOG2 para detectar mudanças no ambiente.
fgbg = cv2.createBackgroundSubtractorMOG2()

# Inicializa o alarme sonoro
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alarm.mp3")

# O programa entra em um loop infinito para processar continuamente os quadros da câmera.
while True:
    ret, frame = cap.read()

    # Aplicar o detector de fundo
    fgmask = fgbg.apply(frame)

    # Aplicar um limite para separar o primeiro plano do fundo
    thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]

    # Encontrar os contornos dos objetos em movimento
    countors, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for countor in countors:
        if cv2.contourArea(countor) > 5000: # limite de área para o movimento

            #Região de Interesse (RoI) da zona de movimento
            # Se o movimento for detectado, marcamos a região de interesse (RoI) 
            # com um retângulo verde no quadro e acionamos o alarme sonoro.

            x, y, w, h = cv2.boundingRect(countor) #  calcula o retângulo delimitador

            # desenhamos um retângulo verde ao redor da região de interesse (RoI) no quadro frame
            # (x, y) representa o ponto de início do retângulo (canto superior esquerdo), e (x + w, y + h) representa o ponto final (canto inferior direito). 
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) #(0, 255, 0), 2)  cor e borda

            #Aqui, estamos criando uma região de interesse (RoI) chamada roi no quadro frame
            roi = frame[y:y + h, x:x + w]

            alert_sound.play()

    cv2.imshow('Detectar de Movimentos', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()