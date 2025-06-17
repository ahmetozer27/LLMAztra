# === utils.py ===

import os
import torch
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(log_file=None):
    """Logging kurulumu"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)


def create_directories(config):
    """Gerekli dizinleri oluştur"""
    directories = [
        os.path.dirname(config['save_path']),
        config['tokenizer_path'],
        os.path.dirname(config['dataset_path_txt']),
        './logs',
        './plots'
    ]

    for directory in directories:
        if directory:  # Boş string kontrolü
            os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Dizin oluşturuldu: {directory}")


def check_gpu():
    """GPU kontrolü ve bilgileri"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9

        logger.info(f"🚀 GPU kullanılabilir!")
        logger.info(f"   GPU sayısı: {gpu_count}")
        logger.info(f"   Aktif GPU: {gpu_name}")
        logger.info(f"   GPU bellek: {gpu_memory:.1f} GB")

        return True, gpu_name, gpu_memory
    else:
        logger.info("⚠️ GPU bulunamadı, CPU kullanılacak")
        return False, None, None


def estimate_training_time(dataset_size, config):
    """Eğitim süresi tahmini"""
    tokens_per_batch = config['batch_size'] * config['block_size']
    batches_per_epoch = dataset_size // tokens_per_batch
    total_batches = batches_per_epoch * config['epochs']

    # GPU/CPU hızına göre tahmin (saniye/batch)
    gpu_available, _, _ = check_gpu()
    time_per_batch = 0.1 if gpu_available else 1.0  # Rough estimate

    estimated_time = total_batches * time_per_batch

    logger.info(f"⏱️ Eğitim süresi tahmini:")
    logger.info(f"   Toplam batch: {total_batches:,}")
    logger.info(f"   Tahmini süre: {estimated_time / 3600:.1f} saat")

    return estimated_time


def plot_training_history(losses, save_path='./plots/training_loss.png'):
    """Eğitim kaybını görselleştir"""
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        logger.info(f"📊 Loss grafiği kaydedildi: {save_path}")
    except Exception as e:
        logger.warning(f"⚠️ Grafik kaydedilemedi: {e}")


def validate_config(config):
    """Config dosyasını doğrula"""
    required_keys = [
        'vocab_size', 'block_size', 'n_layer', 'n_head', 'd_model',
        'batch_size', 'learning_rate', 'epochs'
    ]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config'de eksik anahtar: {key}")

    # Değer kontrolü
    if config['d_model'] % config['n_head'] != 0:
        raise ValueError("d_model, n_head'e tam bölünmelidir")

    if config['block_size'] <= 0:
        raise ValueError("block_size pozitif olmalıdır")

    logger.info("✅ Config doğrulaması başarılı")


def save_config(config, save_path):
    """Config'i kaydet"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"💾 Config kaydedildi: {save_path}")


def load_config(config_path):
    """Config yükle"""
    with open(config_path, 'r') as f:
        return json.load(f)


def count_parameters(model):
    """Model parametrelerini say"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"📊 Model istatistikleri:")
    logger.info(f"   Toplam parametre: {total_params:,}")
    logger.info(f"   Eğitilebilir parametre: {trainable_params:,}")
    logger.info(f"   Model boyutu: ~{total_params * 4 / 1e6:.1f} MB")

    return total_params, trainable_params


def format_time(seconds):
    """Saniyeyi okunabilir formata çevir"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


class TrainingMonitor:
    """Eğitim takip sınıfı"""

    def __init__(self):
        self.losses = []
        self.start_time = None
        self.step_times = []

    def start_training(self):
        self.start_time = datetime.now()
        logger.info(f"🚀 Eğitim başladı: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def log_step(self, step, loss, lr=None):
        self.losses.append(loss)

        if len(self.losses) > 1:
            avg_loss = np.mean(self.losses[-100:])  # Son 100 adımın ortalaması
            logger.info(f"Step {step}: Loss={loss:.4f}, Avg={avg_loss:.4f}")

            if lr:
                logger.info(f"Learning Rate: {lr:.2e}")

    def end_training(self):
        if self.start_time:
            end_time = datetime.now()
            duration = end_time - self.start_time
            logger.info(f"✅ Eğitim tamamlandı: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"⏱️ Toplam süre: {duration}")

    def save_history(self, save_path='./logs/training_history.json'):
        history = {
            'losses': self.losses,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'total_steps': len(self.losses)
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"📜 Eğitim geçmişi kaydedildi: {save_path}")


# === main.py ===

import argparse
import sys
from config import config
from tokenizer.train_tokenizer import main as train_tokenizer_main
from trainer.train import trainModel
from inference.generate import generate_interactive, generate
from utils import *


def main():
    parser = argparse.ArgumentParser(description='Aztra Language Model')
    parser.add_argument('--mode', choices=['train_tokenizer', 'train', 'generate', 'interactive'],
                        required=True, help='Çalıştırma modu')
    parser.add_argument('--prompt', type=str, help='Metin üretimi için prompt')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maksimum token sayısı')
    parser.add_argument('--config', type=str, help='Config dosyası yolu')
    parser.add_argument('--log_file', type=str, help='Log dosyası yolu')
    parser.add_argument('--resume', action='store_true', help='Eğitimi devam ettir')

    args = parser.parse_args()

    # Logging kurulumu
    setup_logging(args.log_file)

    # Config yükle
    if args.config:
        global config
        config = load_config(args.config)

    # Config doğrula
    validate_config(config)

    # Dizinleri oluştur
    create_directories(config)

    # GPU kontrol et
    check_gpu()

    logger.info(f"🎯 Mod: {args.mode}")

    try:
        if args.mode == 'train_tokenizer':
            logger.info("🔤 Tokenizer eğitimi başlıyor...")
            train_tokenizer_main()

        elif args.mode == 'train':
            logger.info("🚂 Model eğitimi başlıyor...")

            # Eğitim süresi tahmini
            if os.path.exists(config['dataset_path_txt']):
                with open(config['dataset_path_txt'], 'r', encoding='utf-8') as f:
                    dataset_size = len(f.read().split())
                estimate_training_time(dataset_size, config)

            trainModel()

        elif args.mode == 'generate':
            if args.prompt:
                logger.info(f"🎨 Metin üretiliyor: '{args.prompt[:50]}...'")
                result = generate(args.prompt, args.max_tokens)
                print(f"\n🎯 Sonuç:\n{result}")
            else:
                logger.error("❌ Generate modu için --prompt gerekli")

        elif args.mode == 'interactive':
            logger.info("💬 İnteraktif mod başlıyor...")
            generate_interactive()

    except KeyboardInterrupt:
        logger.info("⏹️ Kullanıcı tarafından durduruldu")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Hata: {e}")
        sys.exit(1)