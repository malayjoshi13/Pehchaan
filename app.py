# streamlit run app.py

import streamlit as st 
from utils import supported_languages
from services import translate, show_models

translate_button = None
translation_model = None
models = [" "]

# Show user list of supported source and target languages and let them select from the list.
st.title("Language Translation App")
source_language = st.selectbox("Select source language:", supported_languages.final_list)
target_language = st.selectbox("Select target language:", supported_languages.final_list)


# Once user selects both source language and target language.
if source_language != " " and target_language != " ":
    # Let user input source sentence.
    source_text = st.text_area("Enter text to translate:")

    # Show user list of model(s) based on source and target languages
    models = show_models.model_list(source_language, target_language)

    # Let user select translation model
    translation_model = st.selectbox("Select transformer-based translation model:", models)

    # Once user select source lang, target lang, source sentence and translation model, show translation button to him/her.
    if source_text != " " and translation_model != " ":
        translate_button = st.button('Translate')


# Once user clicks translate button, do translation and show output to user on Streamlits's UI
if translate_button:
    translated_output = translate.do_translation(translation_model, source_text, source_language, target_language)
    st.write(translated_output)
