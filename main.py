# Importing dependencies
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import platform
import time
import pathlib
import os
import json
import zipfile

# Versions
print('Python version:', platform.python_version())
print('Tensorflow version:', tf.__version__)
print('Keras version:', tf.keras.__version__)

# Development variables
silent = True
DEBUG = False
DEBUG_EXAMPLES = 10


# -----LOADING DATASET-----
dataset_file_names = [
    'recipes_raw_nosource_ar.json',
    'recipes_raw_nosource_epi.json',
    'recipes_raw_nosource_fn.json',
]

dataset = []

for dataset_file_name in dataset_file_names:
    dataset_file_path = f'datasets/{dataset_file_name}'

    with open(dataset_file_path) as dataset_file:
        json_data_dict = json.load(dataset_file)
        json_data_list = list(json_data_dict.values())
        dict_keys = [key for key in json_data_list[0]]
        dict_keys.sort()
        dataset += json_data_list

        if not silent:
            print(dataset_file_path)
            print('===========================================')
            print('Number of examples: ', len(json_data_list), '\n')
            print('Example object keys:\n', dict_keys, '\n')
            print('Example object:\n', json_data_list[0], '\n')
            print('Required keys:\n')
            print('  title: ', json_data_list[0]['title'], '\n')
            print('  ingredients: ', json_data_list[0]['ingredients'], '\n')
            print('  instructions: ', json_data_list[0]['instructions'])
            print('\n\n')

dataset_raw = dataset

if not silent:
    print('Total number of raw examples: ', len(dataset))
    


# -----PREPROCESSING DATASET-----
# 1. Filtering out incomplete examples
def recipe_validate_required_fields(recipe):
    required_keys = ['title', 'ingredients', 'instructions']
    
    if not recipe:
        return False
    
    for required_key in required_keys:
        if not recipe[required_key]:
            return False
        
        if type(recipe[required_key]) == list and len(recipe[required_key]) == 0:
            return False
    
    return True

dataset_validated = [recipe for recipe in dataset if recipe_validate_required_fields(recipe)]

if not silent:
    print('Dataset size BEFORE validation', len(dataset))
    print('Dataset size AFTER validation', len(dataset_validated))
    print('Number of invalide recipes', len(dataset) - len(dataset_validated))
    
# 2. Converting recipes objects into strings
STOP_WORD_TITLE = '📗 '
STOP_WORD_INGREDIENTS = '\n🥕\n\n'
STOP_WORD_INSTRUCTIONS = '\n📝\n\n'

def recipe_to_string(recipe):
    # Noise in ar.json
    noize_string = 'ADVERTISEMENT'
    
    title = recipe['title']
    ingredients = recipe['ingredients']
    instructions = recipe['instructions'].split('\n')
    
    ingredients_string = ''
    for ingredient in ingredients:
        ingredient = ingredient.replace(noize_string, '')
        if ingredient:
            ingredients_string += f'• {ingredient}\n'
    
    instructions_string = ''
    for instruction in instructions:
        instruction = instruction.replace(noize_string, '')
        if instruction:
            instructions_string += f'▪︎ {instruction}\n'
    
    return f'{STOP_WORD_TITLE}{title}\n{STOP_WORD_INGREDIENTS}{ingredients_string}{STOP_WORD_INSTRUCTIONS}{instructions_string}'

dataset_stringified = [recipe_to_string(recipe) for recipe in dataset_validated]

if not silent:
    print('Stringified dataset size: ', len(dataset_stringified))
    for recipe_index, recipe_string in enumerate(dataset_stringified[:10]):
        print('Recipe #{}\n---------'.format(recipe_index + 1))
        print(recipe_string)
        print('\n')
        
# 3. Filtering out large recipes
recipes_lengths = []
for recipe_text in dataset_stringified:
    recipes_lengths.append(len(recipe_text))
    
plt.hist(recipes_lengths, bins=50)
plt.show()

plt.hist(recipes_lengths, range=(0, 8000), bins=50)
plt.show()

# Looks like a limit of 2000 characters for the recipes will cover 80+% cases.
# We may try to train RNN with this maximum recipe length limit.
MAX_RECIPE_LENGTH = 2000
if DEBUG:
    MAX_RECIPE_LENGTH = 500
    
def filter_recipes_by_length(recipe_test):
    return len(recipe_test) <= MAX_RECIPE_LENGTH

dataset_filtered = [recipe_text for recipe_text in dataset_stringified if filter_recipes_by_length(recipe_text)]

if not silent:
    print('Dataset size BEFORE filtering: ', len(dataset_stringified))
    print('Dataset size AFTER filtering: ', len(dataset_filtered))
    print('Number of eliminated recipes: ', len(dataset_stringified) - len(dataset_filtered))
    
if DEBUG:
    dataset_filtered = dataset_filtered[:DEBUG_EXAMPLES]
    
    
    
# -----CREATING VOCABULARY-----
STOP_SIGN = '␣'

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    char_level=True,
    filters='',
    lower=False,
    split=''
)

tokenizer.fit_on_texts([STOP_SIGN])
tokenizer.fit_on_texts(dataset_filtered)

# Adding +1 to take into account a special unassigned 0 index.
VOCABULARY_SIZE = len(tokenizer.word_counts) + 1

if not silent:
    print('VOCABULARY_SIZE: ', VOCABULARY_SIZE)



# -----VECTORIZING DATASET-----
dataset_vectorized = tokenizer.texts_to_sequences(dataset_filtered)

if not silent:
    print(dataset_vectorized[0][:10], '...')
    
def recipe_sequence_to_string(recipe_sequence):
    recipe_stringified = tokenizer.sequences_to_texts([recipe_sequence])[0]
    recipe_stringified = recipe_stringified.replace('   ', '_').replace(' ', '').replace('_', ' ')
    print(recipe_stringified)
    
if not silent:
    recipe_sequence_to_string(dataset_vectorized[0])
    
# 1. Add padding
dataset_vectorized_padded_without_stops = tf.keras.preprocessing.sequence.pad_sequences(
    dataset_vectorized,
    padding='post',
    truncating='post',
    maxlen=MAX_RECIPE_LENGTH-1,
    value=tokenizer.texts_to_sequences([STOP_SIGN])[0]
)

dataset_vectorized_padded = tf.keras.preprocessing.sequence.pad_sequences(
    dataset_vectorized_padded_without_stops,
    padding='post',
    truncating='post',
    maxlen=MAX_RECIPE_LENGTH+1,
    value=tokenizer.texts_to_sequences([STOP_SIGN])[0]
)

if not silent:
    recipe_sequence_to_string(dataset_vectorized_padded[0])
    
# 2. Split examples on input and target texts
dataset = tf.data.Dataset.from_tensor_slices(dataset_vectorized_padded)

def split_input_target(recipe):
    input_text = recipe[:-1]
    target_text = recipe[1:]
    return input_text, target_text

dataset_targeted = dataset.map(split_input_target)

if not silent:
    for input_example, target_example in dataset_targeted.take(1):
        print('Input sequence size:', repr(len(input_example.numpy())))
        print('Target sequence size:', repr(len(target_example.numpy())))
        print()
        
        input_stringified = tokenizer.sequences_to_texts([input_example.numpy()[:50]])[0]
        target_stringified = tokenizer.sequences_to_texts([target_example.numpy()[:50]])[0]
        
        print('Input:  ', repr(''.join(input_stringified)))
        print('Target: ', repr(''.join(target_stringified)))
        
# 3. Split up the dataset into batches
if DEBUG:
    BATCH_SIZE = DEBUG_EXAMPLES
    SHUFFLE_BUFFER_SIZE = 1
    dataset_train = dataset_targeted \
        .repeat() \
        .batch(BATCH_SIZE, drop_remainder=True)
else:
    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 1000
    dataset_train = dataset_targeted \
        .shuffle(SHUFFLE_BUFFER_SIZE) \
        .batch(BATCH_SIZE, drop_remainder=True) \
        .repeat()

if not silent:
    for input_text, target_text in dataset_train.take(1):
        print('1st batch: input_text:', input_text)
        print()
        print('1st batch: target_text:', target_text)
        


# ----- BUILD THE MODEL -----
vocab_size = VOCABULARY_SIZE
embedding_dim = 256
rnn_units = 1024

def build_model_1(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        batch_input_shape=[batch_size, None]
    ))

    model.add(tf.keras.layers.LSTM(
        units=rnn_units,
        return_sequences=True,
        stateful=True,
        recurrent_initializer=tf.keras.initializers.GlorotNormal()
    ))

    model.add(tf.keras.layers.Dense(vocab_size))
    
    return model
    
model_1 = build_model_1(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)

model_1.summary()
 


# ----- TRAIN THE MODEL -----
def loss(labels, logits):
    entropy = tf.keras.losses.sparse_categorical_crossentropy(
      y_true=labels,
      y_pred=logits,
      from_logits=True
    )
    
    return entropy

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

model_1.compile(
    optimizer=adam_optimizer,
    loss=loss
)

# 2. Configuring checkpoints
checkpoint_dir = 'checkpoints'
def download_latest_checkpoint(zip_only=True):
    latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    latest_checkpoint_name = os.path.split(latest_checkpoint_path)[-1]
    latest_checkpoint_zip_name = latest_checkpoint_name + '.zip'
    
    print('latest_checkpoint_path: ', latest_checkpoint_path)
    print('latest_checkpoint_name: ', latest_checkpoint_name)
    print('---\n')

    print('Checkpoint files:')
    with zipfile.ZipFile(latest_checkpoint_zip_name, mode='w') as zip_obj:
        for folder_name, subfolders, filenames in os.walk(checkpoint_dir):
            for filename in filenames:
                if filename.startswith(latest_checkpoint_name):
                        print('  - ' + filename)
                        file_path = os.path.join(folder_name, filename)
                        zip_obj.write(file_path, os.path.basename(file_path))
    print('---\n')
    print('Zipped to: ', latest_checkpoint_zip_name)

    if not zip_only:
        files.download(latest_checkpoint_zip_name)
        
def model_weights_from_latest_checkpoint(model):
    latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

    if not latest_checkpoint_path:
        print('Latest checkpoint was not found. Using model as is.')
        return model

    print('latest_checkpoint_path: ', latest_checkpoint_path)

    model.load_weights(latest_checkpoint_path)

    return model

def initial_epoch_from_latest_checkpoint():
    latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

    if not latest_checkpoint_path:
        print('Latest checkpoint was not found. Starting from epoch #0')
        return 0

    print('latest_checkpoint_path: ', latest_checkpoint_path)

    latest_checkpoint_name = os.path.split(latest_checkpoint_path)[-1]
    print('latest_checkpoint_name: ', latest_checkpoint_name)

    latest_checkpoint_num = latest_checkpoint_name.split('_')[-1]
    print('latest_checkpoint_num: ', latest_checkpoint_num)

    return int(latest_checkpoint_num)

# def unzip_checkpoint(checkpoint_zip_path):
#     if not os.path.exists(checkpoint_zip_path):
#         print('Cannot find a specified file')
#         return

#     os.makedirs(checkpoint_dir, exist_ok=True)
#     with zipfile.ZipFile(checkpoint_zip_path, 'r') as zip_obj:
#         zip_obj.extractall(checkpoint_dir)

#     %ls -la ./checkpoints
    
# Unzip uploaded checkpoint to checkpoints folder if needed
# unzip_checkpoint('ckpt_10.zip')

# Loading the latest training data from checkpoints if needed.
# model_1 = model_weights_from_latest_checkpoint(model_1)

# Loading weights from H5 file if needed.
# model_1.load_weights('recipe_generation_rnn_batch_64.h5')
    
# 2. Configuring callbacks
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    patience=5,
    monitor='loss',
    restore_best_weights=True,
    verbose=1
)

checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

# 3. Training
# =============================================================================
# INITIAL_EPOCH  = initial_epoch_from_latest_checkpoint()
# EPOCHS_DELTA = 1
# EPOCHS = INITIAL_EPOCH + EPOCHS_DELTA
# STEPS_PER_EPOCH = 1500
# 
# print('\n')
# print('INITIAL_EPOCH:   ', INITIAL_EPOCH)
# print('EPOCHS_DELTA:    ', EPOCHS_DELTA)
# print('EPOCHS:          ', EPOCHS)
# print('STEPS_PER_EPOCH: ', STEPS_PER_EPOCH)
# 
# history_1 = {}
# 
# history_1[INITIAL_EPOCH] = model_1.fit(
#     x=dataset_train,
#     epochs=EPOCHS,
#     steps_per_epoch=STEPS_PER_EPOCH,
#     initial_epoch=INITIAL_EPOCH,
#     callbacks=[
#         checkpoint_callback,
#         early_stopping_callback
#     ]
# )
# 
# model_name = 'recipe_generation_rnn_raw_' + str(INITIAL_EPOCH) + '.h5'
# model_1.save(model_name, save_format='h5')
# 
# download_latest_checkpoint(zip_only=True)
# 
# print(history_1)
# =============================================================================

simplified_batch_size = 1
model_1_simplified = build_model_1(vocab_size, embedding_dim, rnn_units, simplified_batch_size)
model_1_simplified.load_weights('Model.h5')
model_1_simplified.build(tf.TensorShape([simplified_batch_size, None]))

model_1_simplified.summary()
model_1_simplified.save("Model_simplified.h5", save_format='h5')


# num_generate
# - number of characters to generate.
#
# temperature
# - Low temperatures results in more predictable text.
# - Higher temperatures results in more surprising text.
# - Experiment to find the best setting.
def generate_text(model, start_string, num_generate = 1000, temperature=1.0):
    # Evaluation step (generating text using the learned model)
    
    padded_start_string = STOP_WORD_TITLE + start_string

    # Converting our start string to numbers (vectorizing).
    input_indices = np.array(tokenizer.texts_to_sequences([padded_start_string]))

    # Empty string to store our results.
    text_generated = []

    # Here batch size == 1.
    model.reset_states()
    for char_index in range(num_generate):
        predictions = model(input_indices)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # Using a categorical distribution to predict the character returned by the model.
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions,
            num_samples=1
        )[-1,0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state.
        input_indices = tf.expand_dims([predicted_id], 0)
        
        next_character = tokenizer.sequences_to_texts(input_indices.numpy())[0]

        text_generated.append(next_character)

    return (padded_start_string + ''.join(text_generated))


def generate_combinations(model, inp):
    recipe_length = 1000
    temperature = 1.0

    generated_text = generate_text(
        model,
        start_string= inp,
        num_generate = recipe_length,
        temperature= temperature
    )
    print(f'Attempt: "{inp}" + {temperature}')
    print('-----------------------------------')
    print(generated_text)
    print('\n\n')
    return generated_text


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def ingredients_to_string(ingredients):
    # Noise in ar.json
    noize_string = 'ADVERTISEMENT'
    
    ingredients_string = ''
    for ingredient in ingredients:
        ingredient = ingredient.replace(noize_string, '')
        if ingredient:
            ingredients_string += f'{ingredient}\n'
            
    return f'{ingredients_string}'

import copy
find_ds = copy.deepcopy(dataset_validated[0:10000])
for recipe in find_ds:
    recipe["ingredients"] = ingredients_to_string(recipe["ingredients"])
    
    
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#nltk.download('wordnet') # first-time use only
#nltk.download('punkt')
#nltk.download('omw-1.4')
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return nltk.word_tokenize(text.lower().translate(remove_punct_dict))

preprocessed_dataset = []
for recipe in find_ds:
    preprocessed_dataset.append(recipe["ingredients"])
    
TfidfVec  = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
tfidf = TfidfVec.fit_transform(preprocessed_dataset)

def find_recipe(input_ingredients):
    tfidf_inp = TfidfVec.transform([input_ingredients])
    highest_similarity = 0
    most_similar_recipe = None
    
    for index, tfidf_recipe in enumerate(tfidf):
        similarity = cosine_similarity(tfidf_recipe, tfidf_inp)

        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_recipe = find_ds[index]

    return most_similar_recipe

print(find_recipe("vanilla cookies cheese"))
    


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route("/")
def hello():
    return "Welcome to machine learning model APIs! Hahhaa"

@app.route("/generate", methods=['POST'])
def predict():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        out = generate_combinations(model_1_simplified, json["ingredients"])
        return jsonify({"output": out})
    else:
        return 'Content-Type not supported!'
    
@app.route("/find", methods=['POST'])
def find():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        out = find_recipe(json["ingredients"])
        return jsonify({"output": out})
    else:
        return 'Content-Type not supported!'

if __name__ == '__main__':
    app.run()
