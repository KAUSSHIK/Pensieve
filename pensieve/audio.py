import speech_recognition as sr

def transcribe_audio():
    r = sr.Recognizer()
    with sr.Microphone() as speech:
        audio = r.listen(speech)
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return "Could not understand audio"
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return "Could not request results from Google Speech Recognition service"
    except OSError as e:
        if "FLAC conversion utility not available" in str(e):
            print("FLAC conversion utility is not available.")
            print("Please install the FLAC command-line application.")
            print("For macOS, run: brew install flac")
            print("For Linux, run: sudo apt-get install flac")
            print("For Windows, download and install FLAC from the official website.")
            return "FLAC conversion utility is not available"
        else:
            raise e