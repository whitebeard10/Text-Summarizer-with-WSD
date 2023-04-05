from flask import Flask, render_template, request
import spacy
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge import Rouge
import ngram as ngram

app = Flask(__name__)

# Load T5 model and tokenizer
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

nlp = spacy.load("en_core_web_sm")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Take paragraph input
        paragraph = request.form['paragraph']

        # Perform POS tagging and NER using spaCy
        doc = nlp(paragraph)

        # Print POS tagging and NER
        for token in doc:
            print(f"{token.text}: {token.pos_}, {token.ent_type_}")


        # Make list of words that are repeating and/or have different POS tagging in different occurrences
        repeated_words = {}
        for token in doc:
            if token.is_punct or token.text.lower() in ["a", "an", "the"]:
                continue
            if token.text.lower() in repeated_words:
                repeated_words[token.text.lower()].append((token.text, token.pos_, token.ent_type_))
            else:
                repeated_words[token.text.lower()] = [(token.text, token.pos_, token.ent_type_)]

        # Filter out auxiliary verbs from the list of repeating words
        repeated_words = {word: taggings for word, taggings in repeated_words.items() if not all(t[1] == "AUX" for t in taggings)}

        # Make a list of potential target words to perform WSD
        target_words=[]
        for word, taggings in repeated_words.items():
            if len(taggings) > 1:
                print(f"{word}: {[t[1] for t in taggings]}")
                target_words.append((word, [t[1] for t in taggings], [t[2] for t in taggings]))
        

        # Generate summaries using T5 and compare results
        summaries = []
        for word, pos_tags, ner_tags in target_words:
            # Generate input sequence with target word
            input_seq = f"Potential target word: {word}\n\n"
            for token in doc:
                if token.text.lower() == word.lower():
                    input_seq += f"[{word}]"
                else:
                    input_seq += token.text_with_ws

            # Generate summary using T5
            input_ids = t5_tokenizer.encode(input_seq, return_tensors='pt')
            output = t5_model.generate(input_ids, max_length=10000, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

            # Get the top 5 predictions for the target word using an n-gram model
            candidates = repeated_words[word.lower()]
            candidates_str = [c[0] for c in candidates]
            predictions = ngram.NGram(candidates_str).search(word.lower(), threshold=0.5)[:5]
            if len(predictions) > 0:
                top_prediction = predictions[0][0]  # Only use the word from the top prediction
            else:
                top_prediction = word  # If n-gram model cannot find any prediction, keep the original word

            # Replace the target word with the top prediction
            summary = t5_tokenizer.decode(output[0], skip_special_tokens=True)
            summary = summary.replace(word, top_prediction)

            summaries.append(summary)

        # Compare summaries and find the best one
        best_summary = ""
        best_rouge_score = 0
        rouge = Rouge()
        for i, summary in enumerate(summaries):
            score = rouge.get_scores(summary, paragraph)[0]['rouge-1']['f']
            if score > best_rouge_score:
                best_summary = summary
                best_rouge_score = score
            #print the summary and its rouge score
            print(f"Summary {i+1}: {summary}")
            print(f"Rouge score: {score:.2f}\n")
        # Render the results in the template
        return render_template('index.html', paragraph=paragraph, summary=best_summary, rouge_score=best_rouge_score)
    else:
        return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)


