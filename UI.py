import tkinter as tk
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
from subprocess import call

global name
def filename():
    filename.filename = askopenfilename()
    name = filename


main = tk.Tk()
main.title('Report Parser')
canvas = tk.Canvas(main, width = 300, height = 250)  
img = ImageTk.PhotoImage(Image.open("and.jpeg"))  
canvas.create_image(150, 100,image=img)
canvas.grid(row=0,column=0)
button =  tk.Button(main, text='Browse', width=5, command=filename)
button.grid(row=1,column=0)
button2 =  tk.Button(main, text='Start', width=5, command=main.destroy)
button2.grid(row=2,column=0)
#filename = askopenfilename()
main.mainloop()
print("python felz.py --image " + filename.filename)

call("python felz.py --image " + filename.filename, shell = True)




