import tkinter as tk
from tkinter import ttk

def show():

    tempList = [['Jim', '0.33'], ['Dave', '0.67'], ['James', '0.67'], ['Eden', '0.5']]
    tempList.sort(key=lambda e: e[1], reverse=True)

    for i, (name, score) in enumerate(tempList, start=1):
        listBox.insert("", "end", values=(i, name, score))

scores = tk.Tk()
scores.title("Game stats")
label = tk.Label(scores, text="Stats", font=("Arial",30), bg='black', fg='yellow').grid(row=0, columnspan=7)

cols = ('Max Left Lean', 'Max Right Lean', 'Chocolates collected', 'No. of times leaned left', 'No. of times leaned right', 'Time on left', 'Time on right')
listBox = ttk.Treeview(scores, columns=cols, show='headings', style="Custom.Treeview", height=1)
for col in cols:
    listBox.column(col, anchor='center')

for col in cols:
    listBox.heading(col, text=col, anchor='center')
listBox.grid(row=1, column=0, columnspan=2)

tempList = [32.5, 28.5, 4, 4, 6, 13, 12]

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

scores.mainloop()