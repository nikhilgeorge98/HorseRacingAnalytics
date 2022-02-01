import tkinter as tk, tkinter.ttk as ttk
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg as TkAgg
import numpy as np
import pandas as pd
import win32gui
from PIL import ImageGrab
import time
from datetime import date, datetime


def CaptureScreen():
    HWND = win32gui.GetFocus()
    rect=win32gui.GetWindowRect(HWND)
    print(rect)
    x = rect[0]
    x1=x+top.winfo_width()
    y = rect[1]
    y1=y+top.winfo_height()
    im=ImageGrab.grab((x,y,x1,y1))
    im.save("{}.jpeg".format(str(datetime.now()).replace(':', '.')),'jpeg')

data = pd.read_csv('thread_test.csv')
data['sideways angle'] = data['sideways angle'] - 90
data['forward angle'] = data['forward angle'] - 90
table_data = pd.read_csv('stats.csv')

top = tk.Tk()
top.title("Game stats")
top.geometry("{}x{}".format(top.winfo_screenwidth(), top.winfo_screenheight()))
top.configure(bg='black')

## Adding Frame to bundle Treeview with Scrollbar (same idea as Plot+Navbar in same Frame)
tableframe = tk.Frame(top, height=100, bg='black')
tableframe.pack(side='top', fill='y')

label = tk.Label(tableframe, text="Game Stats", font=("Arial",30), bg='black', fg='#148F77').grid(row=0, columnspan=7)

cols = ('Max Left Lean(deg)', 'Max Right Lean(deg)', 'Chocolates collected', 'No. of times leaned left', 'No. of times leaned right', 'Time on left(s)', 'Time on right(s)')
listBox = ttk.Treeview(tableframe, columns=cols, show='headings', style="Custom.Treeview", height=1)
for col in cols:
    listBox.column(col, anchor='center')

for col in cols:
    listBox.heading(col, text=col, anchor='center')
listBox.grid(row=1, column=0, columnspan=2)

# showScores = tk.Button(tableframe, text="Show scores", width=15, command=CaptureScreen).grid(row=4, column=0)
# closeButton = tk.Button(tableframe, text="Close", width=15, command=exit).grid(row=4, column=1)

tempList = table_data.iloc[:,[1]]
# tablelist = [tabledata.iloc[2]]

listBox.insert("", "end", values=tempList)

#********************************** -styles- *****************************************

style = ttk.Style()
style.theme_use('clam')
style.element_create("Custom.Treeheading.border", "from", "default")
style.layout("Custom.Treeview.Heading", [
    ("Custom.Treeheading.cell", {'sticky': 'nswe'}),
    ("Custom.Treeheading.border", {'sticky':'nswe', 'children': [
        ("Custom.Treeheading.padding", {'sticky':'nswe', 'children': [
            ("Custom.Treeheading.image", {'side':'right', 'sticky':''}),
            ("Custom.Treeheading.text", {'sticky':'we'})
        ]})
    ]}),
])
style.configure("Custom.Treeview.Heading",
    background="#16A085", foreground="white", relief="flat", font=(None, 12, 'bold'))
style.map("Custom.Treeview.Heading",
    relief=[('active','groove'),('pressed','sunken')])
style.configure("Treeview", background="#73C6B6", foreground="white", rowheight="50", fieldbackground="#16A085", font=('Calibri', 18,'bold'), padding=10)
style.map("Treeview", background=[('selected', 'yellow')])

#***************************************************************************************

view_nets = tk.Frame(top, bg='black')
view_nets.pack(fill='both', expand=True)

label = tk.Label(view_nets, text="Movement Plots", font=("Arial",30), bg='black', fg='white').pack(side='top')

f, axis = plt.subplots(2, 1)

axis[0].plot(data['overall_update'], data['forward angle'])
axis[0].set_title("Hip range of motion(forward)")
axis[0].set_ylabel("Flexion/Extension(in deg)")

axis[1].plot(data['overall_update'], data['sideways angle'])
axis[1].set_title("Hip range of motion(sideways)")
axis[1].set_ylabel("Flexion/Extension(in deg)")

canvas = TkAgg.FigureCanvasTkAgg(f, master=view_nets)
canvas.draw()
## canvas.get_tk_widget().grid(column = 0, row = 0) I'll explain commenting this out below

toolbar = TkAgg.NavigationToolbar2Tk(canvas, view_nets)
toolbar.update()
canvas._tkcanvas.pack(fill='both',expand=True)

endframe = tk.Frame(top, bg='black')
endframe.pack(side='bottom',fill='y')

label = tk.Label(endframe, text="{}".format(datetime.now()), font=("Arial",20), bg='black', fg='white').grid(row=0, columnspan=6)
showScores = tk.Button(endframe, text="Save as .jpeg", width=20, command=CaptureScreen, bg='green', fg='white').grid(row=0, column=7)

print(tableframe.winfo_screenheight(), view_nets.winfo_screenheight(), top.winfo_height())

top.mainloop()

