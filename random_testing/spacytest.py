import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk
from pycorenlp import StanfordCoreNLP


def spacy_test():
    nlp = spacy.load("de_core_news_lg")
    doc = nlp("Hallo, mein Name ist Hanel, ich wohne an der Oberen Eicherstrasse 37, 87435 Kempten, Allgäu, meine Telefonnummer ist 012345 6789.")
    print(doc.sents)
    print(doc.ents)
    print([(e.label_) for e in doc.ents])
    print(doc)
    print([(w.text, w.pos_) for w in doc])


def nltk_test():

    # Download the necessary NLTK data
    nltk.download("punkt_tab")
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download("maxent_ne_chunker_tab")


    # Define the German text
    text = "Hallo, mein Name ist Hanel, ich wohne an der Oberen Eicherstrasse 37, 87435 Kempten, Allgäu, meine Telefonnummer ist 012345 6789."

    # Tokenize the text
    tokens = word_tokenize(text)
    sentences = sent_tokenize(text)

    # Part-of-speech tagging
    tagged_tokens = pos_tag(tokens)

    # Named entity recognition
    chunked = ne_chunk(tagged_tokens)
    print(chunked)

    # Extract the address, name, zip code, and phone number
    name = None
    address = None
    zipcode = None
    phone = None

    for tree in chunked:
        if hasattr(tree, 'label'):
            if tree.label() == 'PERSON':
                name = ' '.join([leaf[0] for leaf in tree.leaves()])
            elif tree.label() == 'GPE' or tree.label() == 'LOCATION':
                address = ' '.join([leaf[0] for leaf in tree.leaves()])
            elif tree.label() == 'ORGANIZATION':
                pass
            else:
                pass

    for sentence in sentences:
        for word in word_tokenize(sentence):
            if word.isdigit() and len(word) == 5:
                zipcode = word
            elif word.isdigit() and len(word) == 11:
                phone = word

    # Print the extracted information
    print("Name:", name)
    print("Address:", address)
    print("Zipcode:", zipcode)
    print("Phone:", phone)


def core_nlp():
    # Create a Stanford CoreNLP object
    nlp = StanfordCoreNLP('http://localhost:8080')

    # Define the German text
    text = "Hallo, mein Name ist Hanel, ich wohne an der Oberen Eicherstrasse 37, 87435 Kempten, Allgäu, meine Telefonnummer ist 012345 6789."
    # Annotate the text
    output = nlp.annotate(text, properties={
        'annotators': 'tokenize, ssplit, pos, lemma, ner',
        'outputFormat': 'json',
        'lang': 'de'
    })

    # Extract the address, name, zip code, and phone number
    name = None
    address = None
    zipcode = None
    phone = None

    for sentence in output['sentences']:
        for token in sentence['tokens']:
            if token['ner'] == 'PERSON':
                name = token['word']
            elif token['ner'] == 'LOCATION':
                address = token['word']
            elif token['ner'] == 'ORGANIZATION':
                pass
            elif token['word'].isdigit() and len(token['word']) == 5:
                zipcode = token['word']
            elif token['word'].isdigit() and len(token['word']) == 11:
                phone = token['word']

    # Print the extracted information
    print("Name:", name)
    print("Address:", address)
    print("Zipcode:", zipcode)
    print("Phone:", phone)


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_llm():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "Hallo, mein Name ist Hanel, ich wohne an der Oberen Eicherstrasse 37, 87435 Kempten, Allgäu, meine Telefonnummer ist 012345 6789."
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. Your job is to take sentences said by some people and to format them into a json format. You will usually receive a name, their adress with street number, their city the zipcode and their phone number."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)

if __name__ == '__main__':
    test_llm()