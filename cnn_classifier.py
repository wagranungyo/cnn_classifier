import tensorflow_datasets as tfds
import tensorflow_datasets as tfdf

datasetm info = tfds.load("imdb_reviews/subwords8k",with_info = True, as_suprevised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

model = tf.keras.Sequencial([
    tf.keras.layers.Embedding(10000,64),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D()
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
print(tf.__version__)