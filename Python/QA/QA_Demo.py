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

print('Passage:\n', passage )
print (f'Length of the passage: {len(passage.split())} words')

question1 ="What is dipenogoro war" 
print ('\nQuestion 1:\n', question1)
#Getting answer 
_, _ , _ , _, ans  = QA_System.answering_machine ( question1, passage)
print('\nAnswer: ', ans ,  '\n')

question2 ="who lead diponegoro war" 
print ('\nQuestion 2:\n', question2)
#Getting answer 
_, _ , _ , _, ans  = QA_System.answering_machine ( question2, passage)
print('\nAnswer: ', ans ,  '\n')

question3 ="where diponegoro war happen" 
print ('\nQuestion 3:\n', question3)
#Getting answer 
_, _ , _ , _, ans  = QA_System.answering_machine ( question3, passage)
print('\nAnswer: ', ans ,  '\n')

question4 ="when the diponegoro war happen" 
print ('\nQuestion 4:\n', question4)
#Getting answer 
_, _ , _ , _, ans  = QA_System.answering_machine ( question4, passage)
print('\nAnswer: ', ans ,  '\n')

question5 ="why diponegoro war happen" 
print ('\nQuestion 5:\n', question5)
#Getting answer 
_, _ , _ , _, ans  = QA_System.answering_machine ( question5, passage)
print('\nAnswer: ', ans ,  '\n')

question6 ="how diponegoro war happen" 
print ('\nQuestion 6:\n', question6)
#Getting answer 
_, _ , _ , _, ans  = QA_System.answering_machine ( question6, passage)
print('\nAnswer: ', ans ,  '\n')