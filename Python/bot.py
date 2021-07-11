import pyautogui, time, os, random
time.sleep(5)
rand = random.randint(5, 10)
a = ["!points", "makan ayam", "kue bulan", "sampah masyarakat", "itu nael", "2 3 tutup botol", "muka lu kek k....."]
for i in range (1000000000):
    time.sleep(rand)
    pyautogui.typewrite(random.choice(a))
    time.sleep(random.randint(0, 5))
    pyautogui.press("enter")
os.kill()

