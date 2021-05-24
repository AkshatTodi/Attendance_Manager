#!/usr/bin/Python

import tkinter as tk
#from tkinter import *
from gtts import gTTS
from tkinter import Message,Text
import datetime
import time
import shutil
import cv2,os
import csv
#import scipy
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import tkinter.ttk as ttk
import tkinter.font as font

top = tk.Tk()
#Code to add widgets will go here...

top.title('Attendance')
top.configure(background='blue')
#top.attributes('-fullscreen', True)
top.geometry('800x400')

#mytext1 = 'Please enter your details'
#mytext2 = 'I am detecting your face'
#mytext3 = 'Your details and images are combining'
#mytext4_1 = 'Your face is recigised'
#mytext4_2 = 'Your face is unknown'

#language = 'en'

#myobj1 = gTTS(text=mytext1, lang=language, slow=False)
#myobj2 = gTTS(text=mytext2, lang=language, slow=False)
#myobj3 = gTTS(text=mytext3, lang=language, slow=False)
#myobj4 = gTTS(text=mytext4_1, lang=language, slow=False)
#myobj5 = gTTS(text=mytext4_2, lang=language, slow=False)

#myobj1.save("prasence.mp3")
#os.system("prasence.mp3")

#top.grid_rowconfigure(0, weight=1)
#top.grid_columnconfigure(0, weight=1)

def Clear1():
    txt1.delete(0, 'end')
    res =  ""
    txt1.configure(text = res)

def Clear2():
    txt2.delete(0, 'end')
    res = ""
    txt2.configure(text = res)

def Clear3():
    txt3.delete(0, 'end')
    res = ""
    txt3.configure(text = res)

'''def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False'''

def TakeImages():
    #myobj2.save("prasence.mp3")
    #os.system("prasence.mp3")
    Id=(txt1.get())
    name=(txt2.get())
    Year=(txt3.get())
    if(Id.isnumeric() and name.isalpha() and Year.isnumeric()):
        cam = cv2.VideoCapture(0)
        face_cascade = "haarcascade_frontalface_default.xml"
        eye_cascade = "haarcascade_eye.xml"
        smile_cascade = "haarcascade_smile.xml"
        face_detector = cv2.CascadeClassifier(face_cascade)
        eye_detector = cv2.CascadeClassifier(eye_cascade)
        smile_detector = cv2.CascadeClassifier(smile_cascade)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces =face_detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) # cv2.rectangle(variable name in which we read,starting cordinats of point 1,ending cordinates of point 2,color,thickness of the line)
                #roi_gray = gray[y:y+h, x:x+w]
                #roi_color = img[y:y+h, x:x+w]
                eyes = eye_detector.detectMultiScale(gray, 1.3, 20)
                '''for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(img, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
                    smile = smile_detector.detectMultiScale(gray, 1.3, 55)
                    for (sx,sy,sw,sh) in smile:
                        cv2.rectangle(img, (sx,sy), (sx+sw,sy+sh), (0,0,255), 2)'''
                #incrementing sample number
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+Year+'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('frame',img)
            #wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Saved for ID : " + Id + "," + " Name : " + name + "," + " Year : "+ Year
        row = [Id , name, Year]
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message1.configure(text= res)
    else:
        if(Id.isnumeric()):
            res = "Enter Alphabetical Name"
            message1.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message1.configure(text= res)

def TrainImages():
    #myobj3.save("prasence.mp3")
    #os.system("prasence.mp3")
    recognizer = cv2.face_LBPHFaceRecognizer.create() # recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    face_cascade = "haarcascade_frontalface_default.xml"
    eye_cascade = "haarcascade_eye.xml"
    smile_cascade = "haarcascade.xml"
    face_detector = cv2.CascadeClassifier(face_cascade)
    eye_detector = cv2.CascadeClassifier(eye_cascade)
    smile_datector = cv2.CascadeClassifier(smile_cascade)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"#+",".join(str(f) for f in Id)
    message1.configure(text = res)

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #print(imagePaths)

    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        #for (x,y,w,h) in faces:
        faces.append(imageNp)
        Ids.append(Id)
    return faces,Ids

def TrackImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    face_cascade = "haarcascade_frontalface_default.xml"
    eye_cascade = "haarcascade_eye.xml"
    smile_cascade = "haarcascade.xml"
    face_detector = cv2.CascadeClassifier(face_cascade);
    eye_detector = cv2.CascadeClassifier(eye_cascade);
    smile_detector = cv2.CascadeClassifier(smile_cascade);
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX # cv2.FONT_HERSHEY_SIMPLEX is the font in which we want to write the text on out image.
    col_names =  ['Id','Name','Year','Date','Time']
    attendance = pd.DataFrame(columns = col_names) #pd.DataFrame() is alllows us to retrive rows and  columns by positions . In order to do that, we'll need to spcify the positions tof the rows that we want, and the positions of the columns that we want
    while True:
        ret, im =cam.read() # read() method in Python is used to read at most n bytes from the file associated with the given file descriptor. If the end of the file has been reached while reading bytes from the given file descriptor, os. read() method will return an empty bytes object for all bytes left to be read.
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=face_detector.detectMultiScale(gray, 1.3,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            eyes = eye_detector.detectMultiScale(gray, 1.3, 20)
            '''for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(im, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
                smile = smile_detector.detectMultiScale(gray, 1.3, 55)
                for (sx,sy,sw,sh) in smile:
                    cv2.rectangle(im, (sx,sy), (sx+sw,sy+sh), (0,0,255), 2)'''
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            print(conf)
            if(conf < 50):
                #myobj4_1.save("prasence.mp3")
                                    #os.system("prasence.mp3")
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                yy=df.loc[df['Id'] == Id]['Year'].values
                #aa=df.loc[]
                tt=str(Id)+"-"+aa+"-"+str(yy)
                attendance.loc[len(attendance)] = [Id,aa,yy,date,timeStamp]

            else:
                #myobj4_2.save("prasence.mp3")
                #os.system("prasence.mp3")
                Id='Unknown'
                tt=str(Id)
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')
        cv2.imshow('im',im)
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    #print(attendance)
    res=attendance
    message2.configure(text= res)


name = tk.Label(top, text="Face-Recognition-Based-Attendance-Management-System", bg="Green", fg="White", width=45, height=2, font=('times', 20, 'italic bold underline'))
name.place(x=35, y=10)

lbl = tk.Label(top, text="Enter ID :", width=10, height=1, fg="red", bg="yellow", font=('times', 12, ' bold '))
lbl.place(x=70, y=120)

txt1 = tk.Entry(top, width=15, bg="yellow", fg="red", font=('times', 15, ' bold '))
txt1.place(x=195, y=120)

lbl2 = tk.Label(top, text="Enter Name :",width=10 ,fg="red"  ,bg = "yellow" ,height=1 ,font=('times', 12, ' bold '))
lbl2.place(x=450, y=120)

txt2 = tk.Entry(top,width=15  ,bg="yellow"  ,fg="red",font=('times', 15, ' bold ')  )
txt2.place(x=575, y=120)

lbl3 = tk.Label(top, text="Enter Year :",width=10  ,fg="red"  ,bg="yellow"    ,height=1 ,font=('times', 12, ' bold '))
lbl3.place(x=70, y=175)

txt3 = tk.Entry(top,width=15  ,bg="yellow"  ,fg="red",font=('times', 15, ' bold ')  )
txt3.place(x=195, y=175)

lbl4 = tk.Label(top, text="Notification : ",width=10  ,fg="red"  ,bg="yellow"  ,height=1 ,font=('times', 12, ' bold '))
lbl4.place(x=70, y=230)

message1 = tk.Label(top, text="" ,bg="yellow"  ,fg="red"  ,width=30  ,height=1, activebackground = "yellow" ,font=('times', 12, ' bold '))
message1.place(x=195, y=230)

lbl5 = tk.Label(top, text="Attendance : ",width=10  ,fg="red"  ,bg="yellow"  ,height=1 ,font=('times', 12, ' bold '))
lbl5.place(x=70, y=285)

message2 = tk.Label(top, text="" ,fg="red"   ,bg="yellow",activeforeground = "green",width=30  ,height=1  ,font=('times', 12, ' bold '))
message2.place(x=195, y=285)

clearButton1 = tk.Button(top, text="Clear1", command = Clear1, fg="red", bg="yellow", width=10, height=1, activebackground = "Red", font=('times', 12, ' bold '))
clearButton1.place(x=500, y=170)

clearButton2 = tk.Button(top, text="Clear2", command = Clear2, fg="red", bg="yellow", width=10, height=1, activebackground = "Red", font=('times', 12, ' bold '))
clearButton2.place(x=500, y=225)

clearButton3 = tk.Button(top, text="Clear3", command = Clear3, fg="red", bg="yellow", width=10, height=1, activebackground = "Red", font=('times', 12, ' bold '))
clearButton3.place(x=500, y=280)

takeImg = tk.Button(top, text="Take Images", command = TakeImages, fg="red"  ,bg="yellow"  ,width=10  ,height=1, activebackground = "Red" ,font=('times', 12, ' bold '))
takeImg.place(x=630, y=170)

trainImg = tk.Button(top, text="Train Images", command = TrainImages, fg="red"  ,bg="yellow"  ,width=10  ,height=1, activebackground = "Red" ,font=('times', 12, ' bold '))
trainImg.place(x=630, y=225)

trackImg = tk.Button(top, text="Track Images", command = TrackImages, fg="red"  ,bg="yellow"  ,width=10  ,height=1, activebackground = "Red" ,font=('times', 12, ' bold '))
trackImg.place(x=630, y=280)

quitTop = tk.Button(top, text="Quit", command = top.destroy, fg="red"  ,bg="yellow"  ,width=10  ,height=1, activebackground = "Red" ,font=('times', 12, ' bold '))
quitTop.place(x=350, y=335)

#copyWrite = tk.Text(top, background=top.cget("background"), borderwidth=3,font=('times', 12, 'italic bold underline'))
#copyWrite.tag_configure("superscript", offset=10)
#copyWrite.insert("insert","Developed by Akshat","", "TEAM","superscript")
#copyWrite.insert("insert","Developed by Akshat")
#copyWrite.configure(state="disabled",fg="red")
#copyWrite.pack(side="left")
#copyWrite.place(x=650, y=370)
top.mainloop() # mainloop() is used when our application is ready to run. mainloop() is an infinity loop used to run the application, wait for an event to occur and proceed the event as long as the window is not closed.
