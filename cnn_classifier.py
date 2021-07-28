import tensorflow_datasets as tfds
import tensorflow_datasets as tfdf

datasetm info = tfds.load("imdb_reviews/subwords8k",with_info = True, as_suprevised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
print(tf.__version__)