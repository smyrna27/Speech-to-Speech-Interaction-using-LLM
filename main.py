import speech_recognition as sr
import pyttsx3
from transformers import pipeline

# Initialize the recognizer for speech recognition
recognizer = sr.Recognizer()

# Initialize the text-to-speech engine (pyttsx3)
engine = pyttsx3.init()

# You can adjust the speech rate and volume here
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

# Initialize a pre-trained NLP model for text generation
nlp_model = pipeline('text-generation', model='gpt2')

def listen_and_recognize():
    """Capture speech and convert it to text using ASR."""
    with sr.Microphone() as source:
        print("Listening for your input...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source)  # Listen to the microphone input
        try:
            print("Recognizing...")
            text = recognizer.recognize_google(audio)  # Recognize speech using Google API
            print(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None

def process_and_respond(text):
    """Process the recognized text and generate a response using GPT-2."""
    if text:
        print("Processing text...")
        response = nlp_model(text, max_length=50, num_return_sequences=1)[0]['generated_text']
        print(f"Response: {response}")
        return response
    return "Sorry, I couldn't understand your input."

def speak(text):
    """Convert text to speech using pyttsx3."""
    engine.say(text)
    engine.runAndWait()

# Main loop for speech-to-speech interaction
def speech_to_speech():
    while True:
        # Listen to user's speech and recognize it
        recognized_text = listen_and_recognize()

        if recognized_text:
            # Process the recognized text to generate a response
            response = process_and_respond(recognized_text)

            # Speak the response out loud
            speak(response)

if __name__ == "_main_":
    print("Starting Speech-to-Speech System...")
    speech_to_speech()