import speech_recognition as spr #Step  0 - Importing required modules.
from googletrans import Translator, LANGUAGES
from gtts import gTTS
import os
#Step 1 -  Preparing the data set requered
# Creating Recognizer() class object
recog1 = spr.Recognizer()
# Creating microphone instance
mc = spr.Microphone()
# Capture Voice
with mc as source:
    print("Speak 'hello' to initiate the Translation!")
    print("`~!@#$%^&*()-_=+[{]};:',<.>/??/>.<,\':;}]{[+=_-)(*&^%$#@!~`")
    recog1.adjust_for_ambient_noise(source, duration=0.2)
    audio = recog1.listen(source)
    try:
        MyText = recog1.recognize_google(audio)
        MyText = MyText.lower()
        print("Recognized: ",MyText)
    except spr.UnknownValueError:
        print("Sorry, I could not understand the audio. Please speak clearly.")
        MyText = None
    except spr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; ",e)
        MyText = None
#Step 2 - Training the model. uploading the data set into the model.
# If "hello" is detected in the speech
if MyText and 'hello' in MyText:
    # Translator method for translation
    translator = Translator()
    # Source and destination languages
    from_lang = 'en'
    to_lang = 'gu'
    with mc as source:
        print("Speak a sentence...")
        recog1.adjust_for_ambient_noise(source, duration=0.2)
        # Storing the speech into audio variable
        #Step 3 - Uploading the data set into the model.
        audio = recog1.listen(source)
        #Step 4 - Giving the output.
        try:
            # Using recognize_google() method to convert audio into text
            get_sentence = recog1.recognize_google(audio)
            print("Phrase to be Translated: ",get_sentence)
            # Using translate() method for translation
            text_to_translate = translator.translate(get_sentence, src=from_lang, dest=to_lang)
            translated_text = text_to_translate.text
            #Printing the required output in the output language.
            print("Translated text: ",translated_text)
            # Using Google-Text-to-Speech to convert text to speech
            speak = gTTS(text=translated_text, lang=to_lang, slow=True)
            speak.save("captured_voice.mp3")
            # Using OS module to run the translated voice
            os.system("start captured_voice.mp3")
        except spr.UnknownValueError:
            print("Sorry, I could not understand the audio. Please speak clearly.")
        except spr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; ",e)
        except Exception as e:
            print("An error occurred during translation: ",e)
print("Other 107 languages also available. Contact the team for proceding.")


