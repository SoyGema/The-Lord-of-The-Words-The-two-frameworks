

import argparse
from transformers import pipeline


def translation_function(model_name , input_text):
    """Translate text with model """
    translator = pipeline("translation" , model= model_name)
    translated_text = translator(input_text)
    return translated_text


def main():
    parser = argparse.ArgumentParser(description='Translate text using a transformer model.')
    parser.add_argument('--model_name', required=True, help='Name of the translation model')
    parser.add_argument('--input_text', required=True, help='Text to be translated')

    args = parser.parse_args()
    
    translated_result = translation_function(args.model_name, args.input_text)
    print("Translated text:", translated_result[0]['translation_text'])

if __name__ == '__main__':
    main()

