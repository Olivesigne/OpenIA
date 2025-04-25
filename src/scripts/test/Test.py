import torch
import torch.nn as nn
import json

# ========== 🔤 Chargement du vocab ==========
with open("Models/custom_vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

vocab = {ch: i for i, ch in vocab.items()}
inv_vocab = {i: ch for ch, i in vocab.items()}
print(vocab)
print()
print(inv_vocab)

def encoder(txt): return [inv_vocab[c] for c in txt]
def decoder(ids): return ''.join([vocab[i] for i in ids])

# ========== ⚙️ Paramètres du modèle ==========
context_size = 1
embed_size = 64
hidden_size = 128
num_heads = 4
num_layers = 2
dropout = 0.1
max_len = context_size
vocab_size = len(vocab)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 🧠 Modèle GPT ==========
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

# ========== 🔽 Chargement du modèle ==========
model = GPTModel(vocab_size, embed_size, num_heads, num_layers, hidden_size, max_len, dropout).to(device)
model.load_state_dict(torch.load("models/custom_final_10.pth", map_location=device))
model.eval()

# ========== ✨ Génération ==========
def generer_texte(model, start_text, longueur=300):
    input_ids = encoder(start_text)
    print(f"Ta phrase en language IA : {input_ids}")
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(longueur):
            logits = model(input_tensor)
            dernier_logits = logits[:, -1, :]
            probs = torch.softmax(dernier_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

            input_ids.append(next_id)
            input_tensor = torch.tensor([input_ids[-context_size:]], dtype=torch.long).to(device)

    print(f"Sa réponse en language IA : {input_ids}")

    return decoder(input_ids)

# ========== 🧪 Test ==========
texte_de_depart = input("💬 Entrez un texte de départ : ")
print("🧠 Génération en cours...")
texte_genere = generer_texte(model, texte_de_depart, longueur=300)
print("\n📝 Résultat :\n", texte_genere)
