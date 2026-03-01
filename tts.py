import pyttsx3
import threading

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 170)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

def speak_async(text):
    thread = threading.Thread(target=speak, args=(text,))
    thread.daemon = True
    thread.start()