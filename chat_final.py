import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from flask import Flask, request, jsonify
from ASR_TTS_MT import ASR_call, MT_call, TTS_call
from Response import gemini_model
import time

# Load the model with exception handling
start_time = time.time()
try:
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading the SentenceTransformer model: {e}")
    raise
print(f"Model loading time: {time.time() - start_time:.2f} seconds")

# Function to read sentences and corresponding responses from a CSV file
def read_sentences_and_responses(file_path):
    start_time = time.time()
    try:
        df = pd.read_csv(file_path)
        sentences = df.iloc[:, 0].dropna().tolist()  # First column for sentences
        responses = df.iloc[:, 1].dropna().tolist()  # Second column for responses
        print(f"File reading and parsing time: {time.time() - start_time:.2f} seconds")
        return sentences, responses
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_path} is empty or cannot be parsed.")
        raise
    except Exception as e:
        print(f"Unexpected error reading file {file_path}: {e}")
        raise

# Function to vectorize sentences
def vectorize_sentences(sentences):
    start_time = time.time()
    try:
        vectors = model.encode(sentences)
        print(f"Sentence vectorization time: {time.time() - start_time:.2f} seconds")
        return vectors
    except Exception as e:
        print(f"Error encoding sentences: {e}")
        raise

# Function to create and index the vectors using FAISS
def create_faiss_index(vectors):
    start_time = time.time()
    try:
        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
        index.add(vectors)
        print(f"FAISS index creation time: {time.time() - start_time:.2f} seconds")
        return index
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        raise

# Function to perform similarity search
def search_similar_sentences(index, query_vector, sentences, top_k=5):
    start_time = time.time()
    try:
        D, I = index.search(np.array([query_vector]), top_k)
        print(f"Similarity search time: {time.time() - start_time:.2f} seconds")
        return [(sentences[i], i) for i in I[0]]
    except Exception as e:
        print(f"Error during similarity search: {e}")
        raise

# Initialize Flask app
app = Flask(__name__)

# Load data and create FAISS index
file_path = 'Combined.csv'  # Replace with your file path
start_time = time.time()
try:
    sentences, responses = read_sentences_and_responses(file_path)
    vectors = vectorize_sentences(sentences)
    index = create_faiss_index(vectors)
    print(f"Data initialization time: {time.time() - start_time:.2f} seconds")
except Exception as e:
    print(f"Error initializing data: {e}")
    raise

@app.route('/chat', methods=['POST'])
def chat():
    try:
        print("chat")
        start_time = time.time()
        data = request.get_json()
        input_language = data.get('lang')
        audio_file = data.get("audio")
        if not audio_file:
            error_message = "No audio file provided"
            print(error_message)
            tts_out = TTS_call(input_language, error_message)
            response = {
                'asr_out': error_message,
                'base64_tts_audio': tts_out,
                'options': error_message,
                'restart': 'true'
            }
            print(f"Response being sent: {response}")
            return jsonify(response), 400

        # Step 1: Receive voice input and convert to text using ASR
        asr_start_time = time.time()
        user_input = ASR_call(input_language, audio_file)
        print("user_input", user_input)
        print(f"ASR processing time: {time.time() - asr_start_time:.2f} seconds")

        # Step 2: Detect language and translate if necessary
        if input_language != 'en':
            tar_lan = 'en'
            mt_start_time = time.time()
            user_input = MT_call(input_language, tar_lan, user_input)
            print(f"Translation time: {time.time() - mt_start_time:.2f} seconds")

        # Step 3: Perform chat interaction
        vector_start_time = time.time()
        query_vector = model.encode([user_input])[0]
        similar_sentences = search_similar_sentences(index, query_vector, sentences, top_k=4)
        print(f"Query vectorization and search time: {time.time() - vector_start_time:.2f} seconds")

        # Step 4: Emit options
        options = [f"{i + 1}. {sentence}" for i, (sentence, _) in enumerate(similar_sentences)]
        options.append("5. None of the above")

        options_string = "\n".join(options)
        tts_start_time = time.time()
        tts_out = TTS_call(input_language, options_string)
        print(f"TTS processing time: {time.time() - tts_start_time:.2f} seconds")

        response = {
            'asr_out': user_input,
            'base64_tts_audio': tts_out,
            'options': options_string,
            'restart': 'false'
        }
        print(f"Response being sent (excluding tts_out): {{'asr_out': '{user_input}', 'options': '{options_string}', 'restart': 'false'}}")
        return jsonify(response)

    except Exception as e:
        error_message = f"Error in /chat endpoint: {e}"
        un_error = "An unexpected error occurred."
        tts_out = TTS_call(input_language, un_error)
        if input_language != 'en':
            error_message = MT_call('en', input_language, un_error)
        response = {
            'asr_out': error_message,
            'base64_tts_audio': tts_out,
            'options': error_message,
            'restart': 'true'
        }
        print(f"Response being sent (excluding tts_out): {{'asr_out': '{error_message}', 'options': '{error_message}', 'restart': 'true'}}")
        return jsonify(response), 500

@app.route('/respond', methods=['POST'])
def respond():
    try:
        print("respond")
        start_time = time.time()
        data = request.json
        choice_audio = data.get("audio")
        input_language = data.get('lang')

        words_to_numbers = {
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5'
        }

        if not choice_audio:
            error_message = "No audio file provided"
            print(error_message)
            tts_out = TTS_call(input_language, error_message)
            response = {
                'asr_out': error_message,
                'base64_tts_audio': tts_out,
                'options': error_message,
                'restart': 'true'
            }
            print(f"Response being sent (excluding tts_out): {{'asr_out': '', 'options': '{error_message}', 'restart': 'true'}}")
            return jsonify(response), 400

        # Convert the audio input to text (number choice) using ASR
        asr_start_time = time.time()
        choice_text = ASR_call('en', choice_audio)
        print("ASR output:", choice_text)
        print(f"ASR processing time: {time.time() - asr_start_time:.2f} seconds")

        choice_text = choice_text.strip().lower().strip(".,!?")

        # Convert choice text to number
        if choice_text.isdigit():
            choice = int(choice_text)
        else:
            choice = words_to_numbers.get(choice_text)
            if choice is None:
                raise ValueError("Invalid input")
            choice = int(choice)

        # Validate choice
        if choice not in range(1, 6):
            response_message = "Invalid option. Please try again."
        elif choice == 5:
            response_message = "Please rephrase your question."
        else:
            # Perform similarity search and handle the valid choice
            vector_start_time = time.time()
            selected_sentence, idx = search_similar_sentences(index, model.encode([choice_text])[0], sentences, top_k=4)[choice - 1]
            question = selected_sentence
            answer = responses[idx]
            gem_response_start_time = time.time()
            gem_response = gemini_model(question, answer)  # Get response from Gemini API
            response_message = gem_response
            print(f"Gemini model response time: {time.time() - gem_response_start_time:.2f} seconds")

        # Convert response to TTS and possibly translate
        if input_language != 'en':
            mt_start_time = time.time()
            response_message = MT_call('en', input_language, response_message)
            print(f"Translation time: {time.time() - mt_start_time:.2f} seconds")
        tts_start_time = time.time()
        tts_out = TTS_call(input_language, response_message)
        print(f"TTS processing time: {time.time() - tts_start_time:.2f} seconds")

        response = {
            'asr_out': choice_text,
            'base64_tts_audio': tts_out,
            'options': response_message,
            'restart': 'true' if choice == 1 or 2 or 3 or 4 or 5 else 'false'
        }
        print(f"Response being sent (excluding tts_out): {{'asr_out': '{choice_text}', 'options': '{response_message}', 'restart': 'true' if choice == 5 or choice not in range(1, 6) else 'false'}}")
        return jsonify(response)

    except ValueError:
        error_message = "Invalid input. Please enter a number."
        if input_language != 'en':
            error_message = MT_call('en', input_language, error_message)
        tts_out = TTS_call(input_language, error_message)
        response = {
            'asr_out': choice_text,
            'base64_tts_audio': tts_out,
            'options': error_message,
            'restart': 'false'
        }
        print(f"Response being sent (excluding tts_out): {{'asr_out': '{choice_text}', 'options': '{error_message}', 'restart': 'false'}}")
        return jsonify(response)
    except Exception as e:
        error_message = f"Error in /respond endpoint: {e}"
        un_error = "An unexpected error occurred."
        tts_out = TTS_call(input_language, un_error)
        if input_language != 'en':
            error_message = MT_call('en', input_language, un_error)
        response = {
            'asr_out': choice_text,
            'base64_tts_audio': tts_out,
            'options': error_message,
            'restart': 'true'
        }
        print(f"Response being sent (excluding tts_out): {{'asr_out': '{choice_text}', 'options': '{error_message}', 'restart': 'true'}}")
        return jsonify(response), 500

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=0000, debug=False)
    except Exception as e:
        print(f"Error running Flask app: {e}")
        raise
