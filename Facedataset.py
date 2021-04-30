import cv2
import os
haar_cascade="haarcascade_frontalface_default.xml"
dataset="dataset"
name="kans"

path=os.path.join(dataset,name)
if not os.path.isdir(path):
    os.mkdir(path)

(width,height)=(130,100)
face_cascade = cv2.CascadeClassifier(haar_cascade)
cam=cv2.VideoCapture(0)

count=1
while count < 30:
    print(count)
    _,img=cam.read()
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(grayImg,1.3,4)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        face= grayImg[y:y+h,x:x+h]
        faceresize=cv2.resize(face,(width,height))
        cv2.imwrite("%s/%s.jpg" %(path,count),faceresize)
    count+=1
    cv2.imshow("FaceDataset",img)
    key=cv2.waitKey(10)
    if key == 27:
        break
print("Dataset Created Succesfully")
cam.release()
cv2.destroyAllWindows()

