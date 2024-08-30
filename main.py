import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from utils import *
from model import *

TAU = 0.1           # Margin of odds needed to place
BET_SPLIT = 0.05    # Percentage of total cash on single bet
N_FEATURES = 512    # How many features to select from dataset
DROPOUT = 0         # Level of dropout. <=0 for not using
SHUFFLE_DS = True   # Shuffle dataset. End of season is wild and hard to predict
EPOCHS = 250        # Number of epochs to train model on
PRINT_EVERY = 10    # Print loss every N epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.set_printoptions(precision=2)
print(f"Using {device} for PyTorch")

# Create dataset
X_train, y_train, X_test, y_test, odds_test = generate_datasets(N_FEATURES, shuffle_ds=SHUFFLE_DS, random_state=27)
train = DatasetFBM(X_train, np.array(y_train), device)

# Model
model = FBM(X_train.shape[1], 3, dropout=DROPOUT).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
train_loader = DataLoader(dataset=train, batch_size=64, shuffle=True, collate_fn=train.collate_batch)

# Model in train mode
model.train()
history = []

# Training
for epoch in range(EPOCHS):
    # Iterate through the whole dataset, x & y being batches for training
    for x, y in train_loader:
        optimizer.zero_grad()
        predicted_label = model(x)
        loss = criterion(predicted_label, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        history.append(loss.item())
    if (epoch + 1) % PRINT_EVERY == 0:
        print(f"Epoch {epoch+1:4d}: log loss = {sum(history[-PRINT_EVERY:])/PRINT_EVERY:.3f}")

# Show loss history
# plt.plot(history)
# plt.show()

# Evaluation mode, predict odds for test data
model.eval()
with torch.no_grad():
    y_probs = model(torch.tensor(X_test, dtype=torch.float).to(device))
    y_probs = y_probs.cpu().detach().numpy()
    y_odds = np.reciprocal(y_probs, where=y_probs!=0)

# Use predicted odds for evaluation on real odds
evaluate(y_odds, odds_test, y_test, tau=TAU, bet_split=BET_SPLIT)
