import cv2
import os

import tkinter as tk


import Program

os.system("pip install deepface")

def register():
    # Create images folder
    if not os.path.exists(r"Image"):
        os.makedirs(r"Image")

    cam = cv2.VideoCapture(0)
    
    cv2.namedWindow("test")
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cam.read()
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # _image = frame[int(y): int(y + h), int(x):int( x+w), 0:3]
    
        cv2.imshow("test", frame)
    
        if not ret:
            break
        k = cv2.waitKey(1)
    
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            cam.release()
            cv2.destroyAllWindows()
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = str(name.get() + ".png")
            cv2.imwrite(r"Image\\" + img_name, frame)
            print("{} written!".format(img_name))
            cam.release()
            cv2.destroyAllWindows()
            break
    raiseFrame(loginFrame)

# Tkinter
root = tk.Tk()
root.title("Face Login Example")
# Frames
loginFrame = tk.Frame(root)
regFrame = tk.Frame(root)
userMenuFrame = tk.Frame(root)

# Define Frame List
frameList = [loginFrame, regFrame, userMenuFrame]
# Configure all Frames
for frame in frameList:
    frame.grid(row=0, column=0, sticky='news')
    frame.configure(bg='white')


def raiseFrame(frame):
    frame.tkraise()


def regFrameRaiseFrame():
    raiseFrame(regFrame)


def logFrameRaiseFrame():
    Program.Main()


# Tkinter Vars
# Stores user's name when registering
name = tk.StringVar()
# Stores user's name when they have logged in
loggedInUser = tk.StringVar()

tk.Label(loginFrame, text="Face Recognition", font=("Courier", 60), bg="white").grid(row=1, column=1, columnspan=5)
loginButton = tk.Button(loginFrame, text="Expression Recognition", bg="white", font=("Arial", 30), command=logFrameRaiseFrame)
# 
loginButton.grid(row=2, column=5)
regButton = tk.Button(loginFrame, text="Register", bg="white", font=("Arial", 30),command=regFrameRaiseFrame)
# 
regButton.grid(row=2, column=1)

tk.Label(regFrame, text="Register", font=("Courier", 60), bg="white").grid(row=1, column=1, columnspan=5)
tk.Label(regFrame, text="Name: ", font=("Arial", 30), bg="white").grid(row=2, column=1)
nameEntry = tk.Entry(regFrame, textvariable=name, font=("Arial", 30)).grid(row=2, column=2)

registerButton = tk.Button(regFrame, text="Register", bg="white", font=("Arial", 30),command=register)
# 
registerButton.grid(row=3, column=2)

# 


raiseFrame(loginFrame)
root.mainloop()
