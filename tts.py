import pyttsx3

def text_to_speech(text):
    # Initialize pyttsx3 engine for offline TTS
    engine = pyttsx3.init()
    
    # Adjust properties
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 20) # Slightly slower for clarity
    
    print(f"AI: {text}")
    engine.say(text)
    engine.runAndWait()
