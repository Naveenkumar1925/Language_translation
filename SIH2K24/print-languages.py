from googletrans import Translator, LANGUAGES
# Print the number of available languages
print("Total Languages: ",len(LANGUAGES))
i=1
# Print all available languages
for code, lang in LANGUAGES.items():
    print(i,code,":", lang)
    i+=1
