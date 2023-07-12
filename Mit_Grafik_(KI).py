import pygame
import tensorflow as tf
import numpy as np

# Initialize Pygame
pygame.init()

# Set window dimensions
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Set colors
GRAY = (200, 200, 200)
WHITE = (255, 255, 255)

# Set font
FONT_SIZE = 24
font = pygame.font.Font(None, FONT_SIZE)

# Trainingsdaten
train_data = [
    ["What is your name?", "My name is Chatbot."],
    ["How old are you?", "I am 3 years old."],
    ["What is the capital of France?", "The capital of France is Paris."],
    # Add more question-answer pairs here
]

# Vorverarbeitung der Daten
questions = [data[0] for data in train_data]
answers = [data[1] for data in train_data]

# Rest of your code for tokenization and model training...
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(questions)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for question in questions:
    token_list = tokenizer.texts_to_sequences([question])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Modell erstellen
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(150, return_sequences=True)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.GRU(100))
model.add(tf.keras.layers.Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
batch_size = 1000
num_epochs = 100

# Starte das Training
model.fit(xs, ys, epochs=num_epochs,batch_size=batch_size)


# Funktion zum Trainieren der KI mit neuen Daten
def train(new_train_data):
    new_questions = [data[0] for data in new_train_data]
    new_answers = [data[1] for data in new_train_data]

    tokenizer.fit_on_texts(new_questions)
    total_words = len(tokenizer.word_index) + 1

    new_input_sequences = []
    for question in new_questions:
        token_list = tokenizer.texts_to_sequences([question])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            new_input_sequences.append(n_gram_sequence)

    new_input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(new_input_sequences, maxlen=max_sequence_len, padding='pre'))
    new_xs, new_labels = new_input_sequences[:, :-1], new_input_sequences[:, -1]
    new_ys = tf.keras.utils.to_categorical(new_labels, num_classes=total_words)

   


# Funktion zum Generieren einer Antwort
# Funktion zum Generieren einer Antwort
# Variable zum Speichern des gemerkten Satzes


# Funktion zum Generieren einer Antwort
import random

# Funktion zum Generieren einer Antwort
def generate_answer(question, randomness=1, best_num=1, no_d=1, best_word=0.0):
    if not question.endswith('?'):
        question += '?'  # Füge ein Fragezeichen hinzu, falls keins vorhanden ist

    if best_word > 0.0:
        best_answer = None

        for data in train_data:
            if data[0] == question:
                best_answer = data[1]
                break

        if best_answer is not None:
            return best_answer

        time_for_thinking = 5 / (best_word * 10)
        time_for_thinking = max(time_for_thinking, 0.5)

        print("Die KI denkt über die beste Antwort nach...")
      

    possible_answers = [data[1] for data in train_data if data[0] == question]

    if len(possible_answers) > 0:
        return random.choice(possible_answers)

    # Restlicher Code bleibt unverändert



    token_list = tokenizer.texts_to_sequences([question])[0]
    token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

    answer = ''
    current_sequence = token_list.tolist()[0]

    word_count = 0  # Zähler für die Anzahl der Wörter in der Antwort
    word_occurrences = {}  # Wörter und ihre Anzahl der Vorkommen in der Antwort

    for _ in range(max_sequence_len - 1):
        token_list = tf.keras.preprocessing.sequence.pad_sequences([current_sequence], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list)[0]

        if randomness < 1.0:
            predicted_probs = predicted_probs ** (1.0 / randomness)
            predicted_probs = predicted_probs / np.sum(predicted_probs)

        if best_num == 1:
            best_index = np.argmax(predicted_probs)
            if best_index in current_sequence:
                predicted_index = best_index
            else:
                predicted_index = random.choices(range(total_words), weights=predicted_probs, k=1)[0]
        else:
            predicted_index = random.choices(range(total_words), weights=predicted_probs, k=1)[0]

        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                if no_d > 0.0:
                    if word in word_occurrences:
                        word_occurrences[word] += 1
                        if word_occurrences[word] >= int(1 / no_d):
                            return answer.strip()
                    else:
                        word_occurrences[word] = 1

                answer += ' ' + word
                word_count += 1  # Zähler erhöhen
                break

        current_sequence.append(predicted_index)
        current_sequence = current_sequence[1:]

        if word_count >= 150:  # Prüfen, ob die maximale Wortanzahl erreicht ist
            break

    return answer.strip()
# Set up the Pygame window
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("AI Chatbot")
clock = pygame.time.Clock()

# Define the chat input and response areas
input_rect = pygame.Rect(10, WINDOW_HEIGHT - 40, WINDOW_WIDTH - 20, 30)
response_rect = pygame.Rect(10, 10, WINDOW_WIDTH - 20, WINDOW_HEIGHT - 60)

# Variables to store user input and AI response
user_input = ""
response = ""

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                # Process user input and generate AI response
                response = generate_answer(user_input)
                user_input = ""
            else:
                user_input += event.unicode

    # Clear the window
    window.fill(GRAY)

    # Render user input and response text
    input_surface = font.render("Input: " + user_input, True, WHITE)
    response_surface = font.render("Response: " + response, True, WHITE)

    # Blit the text onto the window
    window.blit(input_surface, (input_rect.x + 5, input_rect.y + 5))
    window.blit(response_surface, (response_rect.x + 5, response_rect.y + 5))

    # Draw the input and response areas
    pygame.draw.rect(window, WHITE, input_rect, 2)
    pygame.draw.rect(window, WHITE, response_rect, 2)

    # Update the display
    pygame.display.flip()
    clock.tick(60)

# Quit Pygame
pygame.quit()
