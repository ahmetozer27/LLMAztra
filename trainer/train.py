# === train.py ===

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast
from model.aztra_transformer import TransformerLM
from config import config
import os
import json
from tqdm import tqdm
import logging
import wandb  # Optional: weights & biases için
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# Logging kurulumu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, tokenizer, text_path, block_size, stride_ratio=0.5):
        logger.info(f"Dataset yükleniyor: {text_path}")

        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Dataset dosyası bulunamadı: {text_path}")

        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        if not text.strip():
            raise ValueError("Dataset boş!")

        tokens = tokenizer.encode(text)
        logger.info(f"Toplam token sayısı: {len(tokens)}")

        if len(tokens) < block_size:
            raise ValueError(f"Veri çok küçük! block_size={block_size}, ancak toplam token={len(tokens)}")

        # Stride hesaplama
        stride = max(1, int(block_size * stride_ratio))

        # Input ve target sequences oluştur
        self.inputs = []
        self.targets = []

        for i in range(0, len(tokens) - block_size, stride):
            input_seq = tokens[i:i + block_size]
            target_seq = tokens[i + 1:i + block_size + 1]  # Bir token kaydırılmış

            self.inputs.append(input_seq)
            self.targets.append(target_seq)

        logger.info(f"Dataset hazırlandı. Toplam örnek sayısı: {len(self.inputs)}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.long)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Linear learning rate scheduler with warmup"""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, save_path):
    """Checkpoint kaydetme"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'config': config
    }

    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint kaydedildi: {save_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Checkpoint yükleme"""
    if not os.path.exists(checkpoint_path):
        logger.info("Checkpoint bulunamadı, sıfırdan başlanıyor...")
        return 0, 0

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    step = checkpoint['step']

    logger.info(f"Checkpoint yüklendi: epoch {epoch}, step {step}")
    return epoch, step


def calculate_perplexity(loss):
    """Perplexity hesaplama"""
    return torch.exp(loss).item()


def trainModel():
    # Dizinleri oluştur
    os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
    os.makedirs(config['tokenizer_path'], exist_ok=True)

    # Tokenizer yükle
    tokenizer_path = os.path.join(config['tokenizer_path'], 'tokenizer.json')
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer bulunamadı: {tokenizer_path}")

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    # Özel tokenlar ekle
    special_tokens = {
        'bos_token': '<bos>',
        'eos_token': '<eos>',
        'pad_token': '<pad>',
        'unk_token': '<unk>'
    }
    tokenizer.add_special_tokens(special_tokens)

    logger.info("Tokenizer yüklendi!")

    # Dataset ve DataLoader
    dataset = TextDataset(
        tokenizer,
        config['dataset_path_txt'],
        config['block_size'],
        config['stride_ratio']
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,  # Paralelleştirme için
        pin_memory=True  # GPU transfer hızı için
    )

    # Model oluşturma
    model = TransformerLM(config)

    # Model parametrelerini say
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Toplam parametre: {total_params:,}")
    logger.info(f"Eğitilebilir parametre: {trainable_params:,}")

    # Device ayarı
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info(f"Kullanılan device: {device}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        logger.info(f"GPU Belleği: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Bellek kullanımı: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Optimizer ve scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Toplam adım sayısını hesapla
    total_steps = len(dataloader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        config['warmup_steps'],
        total_steps
    )

    criterion = nn.CrossEntropyLoss()

    # Checkpoint yükle (varsa)
    start_epoch, global_step = load_checkpoint(model, optimizer, scheduler, config['save_path'])

    # Eğitim döngüsü
    model.train()
    best_loss = float('inf')

    for epoch in range(start_epoch, config['epochs']):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config['epochs']}")

        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_val'])

            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()

            # Progress bar güncelle
            current_lr = optimizer.param_groups[0]['lr']
            perplexity = calculate_perplexity(loss)

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ppl': f"{perplexity:.2f}",
                'lr': f"{current_lr:.2e}"
            })

            # Logging
            if global_step % config['log_every'] == 0:
                logger.info(f"Step {global_step}: Loss={loss.item():.4f}, PPL={perplexity:.2f}")

            # Checkpoint kaydet
            if global_step % config['save_every'] == 0:
                checkpoint_path = config['save_path'].replace('.pt', f'_step_{global_step}.pt')
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, loss.item(), checkpoint_path)

        # Epoch sonu
        avg_loss = epoch_loss / len(dataloader)
        avg_perplexity = calculate_perplexity(torch.tensor(avg_loss))

        logger.info(f"Epoch {epoch + 1} tamamlandı:")
        logger.info(f"  Ortalama Loss: {avg_loss:.4f}")
        logger.info(f"  Ortalama Perplexity: {avg_perplexity:.2f}")

        # En iyi modeli kaydet
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = config['save_path'].replace('.pt', '_best.pt')
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, avg_loss, best_model_path)
            logger.info(f"Yeni en iyi model kaydedildi: {best_model_path}")

    # Final model kaydet
    save_checkpoint(model, optimizer, scheduler, config['epochs'] - 1, global_step, avg_loss, config['save_path'])
    logger.info(f"Eğitim tamamlandı! Model kaydedildi: {config['save_path']}")
