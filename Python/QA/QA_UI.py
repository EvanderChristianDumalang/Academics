import warnings
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import QA_System
import os
# making path for my directory
path = r"E:\Kuliah\Github\School\School\Python\QA"

# open path in my directory
os.chdir(path)

# function to open file and return it 
def read_text_file(file_path):
    with open(file_path, 'r', encoding="utf8") as f:
        passage = f.read()
    return passage

# iterate through all file
for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        file_path = f"{path}\{file}"
  
        # call read text file function
        read_text_file(file_path)

# passage for BERT
passage = read_text_file(file_path)

# making windows
Window = tk.Tk() 

# title and size of window
Window.title("Diponegoro QA")
Window.geometry('1250x600')
  
# frame
Frame1 = tk.Frame(Window)
Frame1.pack()

text1=tk.Label(Frame1,text="Diponegoro QA", font="Arial 14 bold")
text1.grid(row=1,column=0,sticky="w")

text2=tk.Label(Frame1,text="Enter the Question:", font="Arial 14 bold")
text2.grid(row=2,column=0,sticky="w")

questiondata =tk.Entry(Frame1, width=75, font="Arial 14 bold")
questiondata.grid(row=2,column=1,sticky="w")

# function to return answer
def Answer():
    # for checking if the question is 5W+1H
    ques = ('what', 'why', 'where', 'when', 'who', 'how')
    # lowercase question
    question = str(questiondata.get()).lower()
    # check if the first word is in ques
    if (question.startswith(ques)):
        _, _ , _ , _, answering  = QA_System.answering_machine (question, passage)
        text3=tk.Label(Frame1,text="Answer:", font="Arial 14 bold")
        text3.grid(row=3,column=0,sticky="nw")

        ans=ScrolledText(Frame1, width= 75, height=20, font="Arial 14 bold")
        ans.grid(row=3,column=1,sticky="w")
        ans.insert(tk.END, answering)
    else:
        text3=tk.Label(Frame1,text="Answer:", font="Arial 14 bold")
        text3.grid(row=3,column=0,sticky="nw")

        ans=ScrolledText(Frame1, width= 75, height=20, font="Arial 14 bold")
        ans.grid(row=3,column=1,sticky="w")
        ans.insert(tk.END, "Use 5W+1H")

# button to run the function
SizeofArray=tk.Button(Frame1,text="Submit", command=Answer, font="Arial 14 bold")
SizeofArray.grid(row=2,column=2,sticky="w") 

Window.mainloop()