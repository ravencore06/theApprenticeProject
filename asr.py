import speech_recognition as sr


def capture_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\nListening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None


def validate_transcription(text):
    if text is None:
        return False, "No speech detected."
    stripped = text.strip()
    if not stripped:
        return False, "Empty transcription."
    if len(stripped) > 500:
        return False, "Input too long."
    if len(stripped) < 2:
        return False, "Input too short."
    return True, None
