import cv2
 
#cargamos la plantilla e inicializamos la webcam:
#face_cascade = cv2.CascadeClassifier('/Users/elianacarrreno/anaconda3/pkgs/opencv-3.2.0-np112py36_0/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')



cap = cv2.VideoCapture(0)
 

while(True):
    #leemos un frame y lo guardamos
    ret, img = cap.read()
 
    #convertimos la imagen a blanco y negro
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    #buscamos las coordenadas de los rostros (si los hay) y
    #guardamos su posicion
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
 
    #Dibujamos un rectangulo en las coordenadas de cada rostro
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
 
    #Mostramos la imagen
    cv2.imshow('img',img)
     
    #con la tecla 'q' salimos del programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()