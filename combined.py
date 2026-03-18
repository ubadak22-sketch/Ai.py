import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
import requests
from typing import Dict, Union, Tuple


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class ShakespeareDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            text: str, 
            characters: int, 
            block_size: int, 
            train: bool=True
        ):
        super(ShakespeareDataset, self).__init__()
        self.text = text
        self.characters = characters
        str_to_int_dict = {s:i for i,s in enumerate(self.characters)}
        int_to_str_dict = {i:s for i,s in enumerate(self.characters)}
        self.encoder = lambda s: [str_to_int_dict[c] for c in s]
        self.decoder = lambda l: ''.join([int_to_str_dict[i] for i in l])
        self.data = torch.tensor(self.encoder(self.text), dtype=torch.long)
        self.block_size = block_size
        self.train = train

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        idx = index
        if self.train:
            idx = torch.randint(len(self.data) - self.block_size, size=(1,))
        X = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        text = self.text[idx:idx+self.block_size]
        sample = {"X": X, "y": y, "text": text}
        return sample

    def __len__(self) -> int:
        if self.train:
            return 5000
        return len(self.data) - self.block_size


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class SingleHead(nn.Module):
    def __init__(
            self, 
            hidden_dim: int, 
            n_embed: int, 
            block_size: int, 
            dropout: float=0.2
        ):
        super(SingleHead, self).__init__()
        self.key = nn.Linear(n_embed, hidden_dim, bias=False)
        self.query = nn.Linear(n_embed, hidden_dim, bias=False)
        self.value = nn.Linear(n_embed, hidden_dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * C**(-0.5)
        masked_weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        masked_probs = F.softmax(masked_weights, dim=-1)
        masked_probs = self.drop(masked_probs)
        v = self.value(x)
        out = masked_probs @ v
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self, 
            num_heads: int, 
            hidden_dim: int, 
            n_embed: int, 
            block_size: int, 
            dropout: float=0.2
        ):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([SingleHead(hidden_dim, n_embed, block_size) for _ in range(self.num_heads)])
        self.project = nn.Linear(n_embed, n_embed)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        out = torch.cat([sh(x) for sh in self.heads], dim=-1)
        out = self.project(out)
        out = self.drop(out)
        return out


class FeedForward(nn.Module):
    def __init__(
            self, 
            n_embed: int, 
            extend_width: int=4, 
            dropout: float=0.2
        ):
        super(FeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_embed, extend_width*n_embed),
            nn.ReLU(),
            nn.Linear(extend_width*n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class TransformerBlock(nn.Module):
    def __init__(
            self, 
            num_heads: int, 
            n_embed: int, 
            block_size: int
        ):
        super(TransformerBlock, self).__init__()
        hidden_dim = n_embed // num_heads
        self.mhsa = MultiHeadSelfAttention(num_heads, hidden_dim, n_embed, block_size) 
        self.feed_forward = FeedForward(n_embed)
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mhsa(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(
            self, 
            vocab_size: int, 
            block_size: int, 
            n_embed: int, 
            num_heads: int, 
            n_layers: int
        ):
        super(GPT, self).__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[TransformerBlock(num_heads, n_embed, block_size) for _ in range(n_layers)],
        )
        self.norm = nn.LayerNorm(n_embed)        
        self.fc = nn.Linear(n_embed, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        token_embeddings = self.embedding(x)
        positional_embedding = self.positional_embedding_table(torch.arange(T, device=x.device))
        token_embeddings = token_embeddings + positional_embedding
        blocks_out = self.blocks(token_embeddings)
        blocks_out = self.norm(blocks_out)
        logits = self.fc(blocks_out)
        logits = logits.reshape(B*T, self.vocab_size)
        return logits

    def generate(self, idx: torch.Tensor, max_tokens: int) -> torch.Tensor:
        t = idx.shape[1]
        for _ in range(max_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self.forward(idx_cond)
            logits = logits.reshape(1, t, -1)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if t < self.block_size:
                t += 1
        return idx


# ─────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────

def download_data(input_file_path: str):
    if not os.path.exists(input_file_path):
        print("Downloading dataset...")
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)
        print("Dataset is downloaded to {}".format(input_file_path))


def return_dataset(
        data_path: int, 
        split: float, 
        block_size: int
    ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    characters = sorted(list(set(text)))
    dataset_len = len(text)
    train_size = int(dataset_len * split)

    train_text = text[:train_size]
    test_text = text[train_size:]
    
    train_set = ShakespeareDataset(train_text, characters, block_size, train=True)
    test_set = ShakespeareDataset(test_text, characters, block_size, train=False)
    return train_set, test_set


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_one_epoch(
        train_loader: torch.utils.data.DataLoader, 
        model: torch.nn.Module, 
        criterion: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        device: str
    ) -> Dict[str, Union[torch.tensor, float]]:
 
    start = time.time()
    model.train()
    losses = torch.zeros(len(train_loader))
    for i, sample in enumerate(train_loader):
        X = sample["X"].to(device)
        y = sample["y"].to(device)
        text = sample["text"]
        logits = model(X)
        loss = criterion(logits, y.view(-1,))
        losses[i] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    time_elapsed = time.time() - start
    train_info = {"loss": torch.mean(losses), "time": time_elapsed}
    return train_info


def test_one_epoch(
        test_loader: torch.utils.data.DataLoader, 
        model: torch.nn.Module, 
        criterion: torch.nn.Module, 
        device: str
    ) -> Dict[str, Union[torch.tensor, float]]:

    start = time.time()
    model.eval()
    losses = torch.zeros(len(test_loader))
    with torch.inference_mode():
        for i, sample in enumerate(test_loader):
            X = sample["X"].to(device)
            y = sample["y"].to(device)
            text = sample["text"]
            logits = model(X)
            loss = criterion(logits, y.view(-1,))
            losses[i] = loss.item()
    time_elapsed = time.time() - start
    test_info = {"loss": torch.mean(losses), "time": time_elapsed}
    return test_info


def generate_text(
        model: torch.nn.Module, 
        train_set: torch.utils.data.Dataset,
        device: str, 
        num_tokens: int
    ):
    idx = torch.zeros((1,1), dtype=torch.long).to(device)
    print(train_set.decoder(model.generate(idx, num_tokens)[0].tolist()))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Config Params
    data_path = "./input.txt"
    load_path = None
    epochs = 50
    block_size = 256
    split = 0.9
    batch_size = 64
    initial_lr = 3e-4
    min_lr = 1e-4
    evaluate_every = 10
    n_embed = 384
    num_heads = 6
    n_layers = 6
    device_id = 0
    checkpoint_dir = "./results/"

    # Download data
    download_data(data_path)

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Read dataset
    train_set, test_set = return_dataset(data_path, split, block_size)

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    # Set device
    device = torch.device('cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu')

    # Create model
    num_chars = len(train_set.characters)
    model = GPT(num_chars, block_size, n_embed, num_heads, n_layers)
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    
    # LR scheduler
    lambda_func = lambda epoch: max(0.99 ** epoch, min_lr / initial_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)

    # Training loop    
    best_val_loss = 1e5
    for e in range(epochs):
        train_info = train_one_epoch(train_dataloader, model, criterion, optimizer, scheduler, device)
        print("At epoch: {}, train loss: {:.2f}, in {:.2f} seconds".format(e+1, train_info["loss"], train_info["time"]))
        if (e+1) % evaluate_every == 0:
            test_info = test_one_epoch(test_dataloader, model, criterion, device)
            print("\nAt epoch: {}, test loss: {:.2f}, in {:.2f} seconds\n".format(e+1, test_info["loss"], test_info["time"]))
            if best_val_loss > test_info["loss"]:
                torch.save(model.state_dict(), checkpoint_dir + "model_epoch_{}_loss_{:.2f}.pt".format(e, test_info["loss"]))
                best_val_loss = test_info["loss"]

    # Generate some text
    generate_text(model, train_set, device, 500)
