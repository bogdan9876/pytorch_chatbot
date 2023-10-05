import random
import json
import torch
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

all_words = []
tags = []
patterns = []
responses = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        words = nltk.word_tokenize(pattern)
        words = [WordNetLemmatizer().lemmatize(word.lower()) for word in words]
        all_words.extend(words)
        patterns.append(words)
        responses.append(intent['responses'])
        tags.append(intent['tag'])

all_words = sorted(list(set(all_words)))

X_train = []
y_train = []

for idx, pattern_words in enumerate(patterns):
    bag = [1 if word in pattern_words else 0 for word in all_words]
    X_train.append(bag)
    label = tags[idx]
    y_train.append(tags.index(label))

X_train = np.array(X_train)
y_train = np.array(y_train)

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

input_size = len(all_words)
hidden_size = 8
output_size = len(tags)

model = NeuralNet(input_size, hidden_size, output_size)

X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'chatbot_model.pth')

def get_response(intents, model, all_words):
    while True:
        user_input = input("You: ").lower()
        if user_input == "quit":
            break

        user_input_words = nltk.word_tokenize(user_input)
        user_input_words = [WordNetLemmatizer().lemmatize(word.lower()) for word in user_input_words]

        user_input_bag = [1 if word in user_input_words else 0 for word in all_words]

        user_input_tensor = torch.FloatTensor(user_input_bag).unsqueeze(0)
        model.eval()
        output = model(user_input_tensor)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        for intent in intents['intents']:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                print(f"ChatBot: {response}")

get_response(intents, model, all_words)
