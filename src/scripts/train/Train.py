import json
import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

texte = open("wikipediaIA.txt", "r", encoding="utf-8").read()

caracteres = sorted(list(set(texte)))
caracteres.insert(0, "<EOS>")

vocab = {ch: i for i, ch in enumerate(caracteres)}
inv_vocab = {i: ch for ch, i in vocab.items()}

# Encodage et décodage
def encoder(txt): return [vocab[c] for c in txt]
def decoder(ids): return ''.join([inv_vocab[i] for i in ids])

context_size = 1
data = [(encoder(texte[i:i+context_size]), vocab[texte[i + context_size]]) for i in range(len(texte) - context_size)]

# Paramètres customs
batch = 128
base = 0
epochs = 100
num_heads = 4
num_layers = 2
hidden_size = 128
embed_size = 64
max_len = context_size
dropout = 0.1

# Conversion en tensors
X = torch.tensor([x for x, _ in data])
Y = torch.tensor([y for _, y in data])
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)

# Modèle
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_size, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_size, embed_size)
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, ff_hidden_size, max_len, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_size))
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_size, num_heads, ff_hidden_size, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.size()
        x = self.token_embed(x) + self.pos_embed[:, :seq_len]
        x = self.blocks(x)
        return self.fc_out(x)

# Configuration du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Utilisation de : {device}")

vocab_size = len(vocab)

model = GPTModel(vocab_size, embed_size, num_heads, num_layers, hidden_size, max_len, dropout).to(device)
model.to("cuda")
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
os.makedirs("models", exist_ok=True)

print("Début de l'entraînement...")

for epoch in range(epochs):
    model.train()
    total_loss = 0

    print("La \"boucle\" 😢 !")
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        logits = logits[:, -1, :]
        loss = loss_fn(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 50 == 0:
        print(f"\n📘 Époque {epoch+1} | Perte moyenne : {total_loss / len(dataloader):.4f}")

    if epochs % 25 == 0:
        torch.save(model.state_dict(), f"Models/custom_epoch_{epoch+base+1}.pth")
        print(f"🗂️ Modèle enregistré à l'époque {epoch+1}")

with open("Models/custom_vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=4)

torch.save(model.state_dict(), f"Models/custom_final_{epochs+base}.pth")
print("🗂️ Modèle final enregistré")
