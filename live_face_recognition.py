import cv2
import face_recognition
import numpy as np
import os

path='images_for_recognition'
images=[]
classNames=[]
mylist=os.listdir(path)
#print(mylist)
for cl in mylist:#this will append all in a classNames[] the images available in path folder
     curImg=cv2.imread(f'{path}/{cl}')
     images.append(curImg)
     classNames.append(os.path.splitext(cl)[0])
print(classNames)

def find_Encodings(images):#this function is find the faece encodings for all images that we append in classNames[]
     encodeList=[]
     for img in images:
         img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
         encode=face_recognition.face_encodings(img)[0]
         encodeList.append(encode)
     return encodeList
encodeListKnown=find_Encodings(images)
print('Encodings complate.........')

def mainbody():#this is main body for video and live face recognition 
     while True:
          succes,img=cap.read()
          imgs=cv2.resize(img,(0,0),None,0.25,0.25)
          imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
          facesCurFrame=face_recognition.face_locations(imgs)
          encodesCurFrame=face_recognition.face_encodings(imgs,facesCurFrame)
          font =cv2.FONT_HERSHEY_SIMPLEX
          for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
               matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
               faceDist=face_recognition.face_distance(encodeListKnown,encodeFace)
               #print(faceDist)
               matchIndex=np.argmin(faceDist)
               if matches[matchIndex]:
                    name=classNames[matchIndex].upper()
                    #print(name)
                    y1,x2,y2,x1=faceLoc
                    y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
                    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
                    cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
               else:
                    y1,x2,y2,x1=faceLoc
                    y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
                    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
                    cv2.putText(img,'Unknown person',(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

          cv2.imshow('Webcam',img)
          key=cv2.waitKey(1) & 0xFF
          if key == ord('q'):
               break
     cap.release()
     cv2.destroyAllWindows()

while True:#for time saving we can use this same trained model for diifrent recognitions
     print("\nWhat do you want to recognize from image/video/live:")
     print("Enter 1 for image ,2 for video and 3 for live and 0 for exit\n:")
     option=int(input("Enter you option-->"))
     if option==1:
          path=input("Enter your image path-->")
          img=cv2.imread(path)
          imgrs=cv2.resize(img,(0,0),None,0.50,0.50)
          imgs=cv2.cvtColor(imgrs,cv2.COLOR_BGR2RGB)
          facesCurFrame=face_recognition.face_locations(imgs)
          encodesCurFrame=face_recognition.face_encodings(imgs,facesCurFrame)
          for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
               matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
               faceDist=face_recognition.face_distance(encodeListKnown,encodeFace)
               #print(faceDist)
               matchIndex=np.argmin(faceDist)
               if matches[matchIndex]:
                    name=classNames[matchIndex].upper()
                    print(name)
                    cv2.putText(imgs,name,(60,20),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
               else:
                    cv2.putText(imgs,'Unknown person',(60,20),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
                    print('Unknown person')
          cv2.imshow('Tested Image',imgs)
          cv2.waitKey(1)
          input()#for hold screen
          cv2.destroyAllWindows()
     elif option==2:
          path=input("Enter your video path-->")
          cap=cv2.VideoCapture(path)
          mainbody()
     elif option==3:
          cap=cv2.VideoCapture(0)
          mainbody()
     elif option!=0 and option!=1 and option!=2 and option!=3:
          print("Enter right option!!!!!")
     if option==0:
          break
