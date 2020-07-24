import tkinter
import tkinter.filedialog
from face_detection import detectFaces, compareFaces


filePath = ""


root = tkinter.Tk()
root.minsize(800, 600)
frame = tkinter.Frame(root, bg = "#0b0c10")
frame.place(relx = 0, rely = 0, relwidth = 1, relheight = 1)


def btnLoadFileIsClicked():
    global filePath
    filePath = tkinter.filedialog.askopenfilename()

def btnDetectIsClicked():
    global filePath
    if(filePath != ""):
        detectFaces(filePath)

def btnCompareIsClicked():
    compareFaces("")

def btnCompareUserPicturesIsClicked():
    path = tkinter.filedialog.askdirectory(title = 'Select Folder')
    compareFaces(path)


btnLoadFile = tkinter.Button(frame, text = "Load file", bg = "#1f2833", command = btnLoadFileIsClicked, fg = "#ffffff", activebackground = "#45a29e")

btnDetect = tkinter.Button(frame, text = "Detect faces", bg = "#1f2833", command = btnDetectIsClicked, fg = "#ffffff", activebackground = "#45a29e")

btnCompare = tkinter.Button(frame, text = "Compare saved faces", bg = "#1f2833", command = btnCompareIsClicked, fg = "#ffffff", activebackground = "#45a29e")

btnCompareUserPictures = tkinter.Button(frame, text = "Compare faces in your chosen folder", bg = "#1f2833", command = btnCompareUserPicturesIsClicked, fg = "#ffffff", activebackground = "#45a29e")


def drawHomeScreen():
    btnLoadFile.place(relx = 0.35, rely = 0.15, relwidth = 0.3, relheight = 0.1)
    btnDetect.place(relx = 0.35, rely = 0.35, relwidth = 0.3, relheight = 0.1)
    btnCompare.place(relx = 0.35, rely = 0.55, relwidth = 0.3, relheight = 0.1)
    btnCompareUserPictures.place(relx = 0.35, rely = 0.75, relwidth = 0.3, relheight = 0.1)
    root.mainloop()


drawHomeScreen()