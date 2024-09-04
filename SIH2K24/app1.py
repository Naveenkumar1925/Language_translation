import speech_recognition as spr  # Step 0 - Importing required modules.
from googletrans import Translator, LANGUAGES
from gtts import gTTS
import os

# Initialize Translator
translator = Translator()
from_lang = 'en'
to_lang = 'gu'

print("Other 107 languages also available. Contact the team for proceeding.")
print("Your Position:")
print("1.) Student")
print("2.) Teacher\n")
category = int(input("Choose your category: "))

if category == 1:  # Student
    # Placeholder for student-specific functionality
    pass

elif category == 2:  # Teacher
    print("Enter your choice: \n")
    print("1.) English")
    print("2.) Gujarati\n")
    teacher = int(input("Choose language for translation: "))

    if teacher == 1:  # English -> Gujarati
        print("Input type:\n")
        print("1.) Keyboard")
        print("2.) Voice\n")
        teacher_input_type = int(input("Choose input type: "))

        if teacher_input_type == 1:  # Keyboard
            get_sentence = input("Enter the required input in \"English\": ")
            text_to_translate = translator.translate(get_sentence, src=from_lang, dest=to_lang)
            translated_text = text_to_translate.text
            print("Translated text: ", translated_text)
            # Convert translated text to speech
            speak = gTTS(text=translated_text, lang=to_lang, slow=True)
            speak.save("captured_voice.mp3")
            # Play the translated voice
            os.system("start captured_voice.mp3")

        elif teacher_input_type == 2:  # Mic
            # Create Recognizer instance
            recog1 = spr.Recognizer()
            # Create Microphone instance
            mc = spr.Microphone()

            # Capture voice
            with mc as source:
                # Initialize Translator
                translator = Translator()
                from_lang = 'en'
                to_lang = 'gu'
                print("Speak 'hello' to initiate the Translation!")
                recog1.adjust_for_ambient_noise(source, duration=0.2)
                audio = recog1.listen(source)
                try:
                    MyText = recog1.recognize_google(audio)
                    MyText = MyText.lower()
                    print("Recognized: ", MyText)
                except spr.UnknownValueError:
                    print("Sorry, I could not understand the audio. Please speak clearly.")
                    MyText = None
                except spr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; ", e)
                    MyText = None

            if MyText and 'hello' in MyText:
                with mc as source:
                    print("Speak a sentence...")
                    recog1.adjust_for_ambient_noise(source, duration=0.2)
                    audio = recog1.listen(source)
                    try:
                        get_sentence = recog1.recognize_google(audio)
                        print("Phrase to be Translated: ", get_sentence)
                        text_to_translate = translator.translate(get_sentence, src=from_lang, dest=to_lang)
                        translated_text = text_to_translate.text
                        print("Translated text: ", translated_text)
                        speak = gTTS(text=translated_text, lang=to_lang, slow=True)
                        speak.save("captured_voice.mp3")
                        os.system("start captured_voice.mp3")
                    except spr.UnknownValueError:
                        print("Sorry, I could not understand the audio. Please speak clearly.")
                    except spr.RequestError as e:
                        print("Could not request results from Google Speech Recognition service; ", e)
                    except Exception as e:
                        print("An error occurred during translation: ", e)

    elif teacher == 2:  # Gujarati -> Gujarati
        print("Input type:\n")
        print("1.) Keyboard")
        print("2.) Voice\n")
        teacher_input_type = int(input("Choose input type: "))

        if teacher_input_type == 1:  # Keyboard
            translated_text = input("Enter the required input in \"Gujarati\": ")
            print("Translated text: ", translated_text)
            speak = gTTS(text=translated_text, lang=to_lang, slow=True)
            speak.save("captured_voice.mp3")
            os.system("start captured_voice.mp3")

        elif teacher_input_type == 2:  # Mic
            recog1 = spr.Recognizer()
            mc = spr.Microphone()

            with mc as source:
                print("Speak 'hello' to initiate the Translation!")
                recog1.adjust_for_ambient_noise(source, duration=0.2)
                audio = recog1.listen(source)
                try:
                    MyText = recog1.recognize_google(audio)
                    MyText = MyText.lower()
                    print("Recognized: ", MyText)
                except spr.UnknownValueError:
                    print("Sorry, I could not understand the audio. Please speak clearly.")
                    MyText = None
                except spr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; ", e)
                    MyText = None

            if MyText and 'hello' in MyText:
                with mc as source:
                    print("Speak a sentence...")
                    recog1.adjust_for_ambient_noise(source, duration=0.2)
                    audio = recog1.listen(source)
                    try:
                        translated_text = recog1.recognize_google(audio)
                        print("Phrase to be Translated: ", translated_text)
                        speak = gTTS(text=translated_text, lang=to_lang, slow=True)
                        speak.save("captured_voice.mp3")
                        os.system("start captured_voice.mp3")
                    except spr.UnknownValueError:
                        print("Sorry, I could not understand the audio. Please speak clearly.")
                    except spr.RequestError as e:
                        print("Could not request results from Google Speech Recognition service; ", e)
                    except Exception as e:
                        print("An error occurred during translation: ", e)

    else:
        print("Oops!! Invalid input - Please try again.")

else:
    print("Oops!! Invalid input - Please try again.")

    
