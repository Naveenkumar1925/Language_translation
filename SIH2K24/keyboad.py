import speech_recognition as spr #Step  0 - Importing required modules.
from googletrans import Translator, LANGUAGES
from gtts import gTTS
import os
# Translator method for translation
translator = Translator()
# Source and destination languages
from_lang = 'en'
to_lang = 'gu'
get_sentence = input("Enter the required input in \"English\" : ")
text_to_translate = translator.translate(get_sentence, src=from_lang, dest=to_lang)
translated_text = text_to_translate.text
#Printing the required output in the output language.
print("Translated text: ",translated_text)
# Using Google-Text-to-Speech to convert text to speech
speak = gTTS(text=translated_text, lang=to_lang, slow=True)
speak.save("captured_voice.mp3")
# Using OS module to run the translated voice
os.system("start captured_voice.mp3")

