import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from datasets import load_dataset

model = TFAutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

inputs = tokenizer(['Hello world'], padding=True, truncation=True,
                  return_tensors='tf')

output = model(inputs)
print(output)

emotions = load_dataset('SetFit/emotion')
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

emotions_encoded.set_format('tf', 
                            columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])

BATCH_SIZE = 64

def order(inp):
    data = list(inp.values())
    return {
        'input_ids': data[1],
        'attention_mask': data[2],
        'token_type_ids': data[3]
    }, data[0]

train_dataset = tf.data.Dataset.from_tensor_slices(emotions_encoded['train'][:])

train_dataset = train_dataset.batch(BATCH_SIZE).shuffle(1000)

train_dataset = train_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices(emotions_encoded['test'][:])
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

inp, out = next(iter(train_dataset))
print(inp, '\n\n', out)

class BERTForClassification(tf.keras.Model):
    
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs):
        x = self.bert(inputs)[1]
        return self.fc(x)

classifier = BERTForClassification(model, num_classes=6)

classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

history = classifier.fit(
    train_dataset,
    epochs=3
)

classifier.evaluate(test_dataset)