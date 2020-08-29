import cv2
            
def facecrop(image):  
    facedata = "haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(facedata)
    img = cv2.imread(image)
    try:
        minisize = (img.shape[1],img.shape[0])
        miniframe = cv2.resize(img, minisize)
        faces = cascade.detectMultiScale(miniframe)
        for f in faces:
            x, y, w, h = [ v for v in f ]
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            sub_face = img[y:y+h, x:x+w]
            cv2.imwrite('photo.jpg', sub_face)
    except Exception as e:
        print(e)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("Take Photo (Press Space)",frame)
    if cv2.waitKey(1) == 32:
        filename = "photo.jpg"
        cv2.imwrite(filename,frame)
        cv2.imshow("Photo", frame)
        break

cap.release()
cv2.destroyAllWindows()

facecrop('photo.jpg')