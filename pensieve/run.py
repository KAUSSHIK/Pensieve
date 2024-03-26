from audio import transcribe_audio

if __name__ == "__main__":
    print("Starting audio transcription test...")
    transcribed_text = transcribe_audio()
    print(f"Transcribed text: {transcribed_text}")