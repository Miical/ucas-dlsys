import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import tqdm

class Config:
    embedding_dim = 256
    hidden_dim = 512
    lr = 0.001
    max_gen_len = 100
    batch_size = 64
    num_layers = 2
    dropout = 0.3

def prepare_data():
    datas = np.load("tang.npz", allow_pickle=True)
    data = datas["data"]
    ix2word = datas["ix2word"].item()
    word2ix = datas["word2ix"].item()
    data = torch.from_numpy(data).long()
    dataloader = DataLoader(data, batch_size=Config.batch_size, shuffle=True, num_workers=2)
    return dataloader, ix2word, word2ix

class ImprovedPoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(ImprovedPoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        embeds = self.embeddings(input)
        batch_size, seq_len = input.size()
        if hidden is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(input.device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(input.device)
        else:
            h_0, c_0 = hidden
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.norm(output)
        output = self.dropout(output)
        output = self.linear(output)
        return output.view(-1, output.size(-1)), hidden


def generate_poem(model, start_words, ix2word, word2ix):
    results = list(start_words)
    input = torch.tensor([[word2ix["<START>"]]]).long()
    hidden = None
    model.eval()
    with torch.no_grad():
        for i in range(Config.max_gen_len):
            output, hidden = model(input, hidden)
            if i < len(start_words):
                w = results[i]
                input = torch.tensor([[word2ix[w]]]).long()
            else:
                top_index = output[0].topk(1)[1].item()
                w = ix2word[top_index]
                results.append(w)
                input = torch.tensor([[top_index]]).long()
            if w == "<EOP>":
                results.pop()
                break
    return "".join(results)

def train_model(dataloader, ix2word, word2ix):
    model = ImprovedPoetryModel(len(word2ix), Config.embedding_dim, Config.hidden_dim,
                                 num_layers=Config.num_layers, dropout=Config.dropout)
    model.load_state_dict(torch.load("improved_poetry_model.pth"))
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(10, 200):
        total_loss = 0
        for batch in tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            input = batch[:, :-1]
            target = batch[:, 1:].reshape(-1)
            output, _ = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), f"improved_poetry_model_epoch_{epoch+1}.pth")
        print(generate_poem(model, start_words="湖光秋月两相和", ix2word=ix2word, word2ix=word2ix))
    return model


if __name__ == "__main__":
    dataloader, ix2word, word2ix = prepare_data()

    model = train_model(dataloader, ix2word, word2ix)

    poem = generate_poem(model, start_words="湖光秋月两相和", ix2word=ix2word, word2ix=word2ix)
    print("\n生成的诗句：")
    print(poem)
