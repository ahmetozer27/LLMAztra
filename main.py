# === main.py ===

import argparse
import sys
import os
import logging
from config import config
from tokenizer.train_tokenizer import main as train_tokenizer_main
from trainer.train import trainModel
from inference.generate import TextGenerator
from utils import setup_logging, create_directories, validate_config, check_gpu, estimate_training_time

# Logging kurulumu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_text(prompt, max_tokens=100, temperature=0.8, top_k=50, top_p=0.95):
    """Tek seferlik metin üretimi"""
    try:
        generator = TextGenerator()
        result = generator.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        return result
    except Exception as e:
        logger.error(f"Metin üretimi hatası: {e}")
        return None


def generate_interactive():
    """İnteraktif metin üretimi"""
    try:
        generator = TextGenerator()
        logger.info("🎉 İnteraktif mod başlatıldı!")
        logger.info("Çıkmak için 'quit' yazın.")

        while True:
            try:
                prompt = input("\n💭 Prompt girin: ").strip()

                if prompt.lower() in ['quit', 'exit', 'q']:
                    logger.info("👋 Görüşürüz!")
                    break

                if not prompt:
                    continue

                print("🤖 Üretiliyor...")
                result = generator.generate(
                    prompt=prompt,
                    max_new_tokens=100,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95
                )

                print(f"\n📝 Sonuç:\n{result}\n")

            except KeyboardInterrupt:
                logger.info("\n👋 İnteraktif mod sonlandırıldı")
                break
            except Exception as e:
                logger.error(f"Hata: {e}")
                continue

    except Exception as e:
        logger.error(f"İnteraktif mod başlatılamadı: {e}")


def check_requirements():
    """Gerekli dosyaları kontrol et"""
    issues = []

    # Config kontrolü
    try:
        validate_config(config)
    except Exception as e:
        issues.append(f"Config hatası: {e}")

    # Dizin kontrolü
    required_dirs = [
        os.path.dirname(config['save_path']),
        config['tokenizer_path'],
        os.path.dirname(config['dataset_path_txt'])
    ]

    for dir_path in required_dirs:
        if dir_path and not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"📁 Dizin oluşturuldu: {dir_path}")
            except Exception as e:
                issues.append(f"Dizin oluşturulamadı {dir_path}: {e}")

    return issues


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(
        description='Aztra Language Model - Türkçe dil modeli eğitimi ve kullanımı',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Kullanım örnekleri:
  python main.py --mode train_tokenizer        # Tokenizer eğitimi
  python main.py --mode train                  # Model eğitimi
  python main.py --mode generate --prompt "Merhaba"  # Tek metin üretimi
  python main.py --mode interactive            # İnteraktif mod
        """
    )

    # Ana parametreler
    parser.add_argument(
        '--mode',
        choices=['train_tokenizer', 'train', 'generate', 'interactive', 'check'],
        required=True,
        help='Çalıştırma modu'
    )

    # Metin üretimi parametreleri
    parser.add_argument('--prompt', type=str, help='Metin üretimi için başlangıç metni')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maksimum token sayısı')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling sıcaklığı (0.1-2.0)')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.95, help='Nucleus sampling')

    # Sistem parametreleri
    parser.add_argument('--log_file', type=str, help='Log dosyası yolu')
    parser.add_argument('--resume', action='store_true', help='Eğitimi devam ettir')
    parser.add_argument('--verbose', '-v', action='store_true', help='Detaylı çıktı')

    args = parser.parse_args()

    # Verbose mod
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Logging kurulumu
    if args.log_file:
        setup_logging(args.log_file)

    # Başlangıç mesajı
    logger.info("=" * 60)
    logger.info("🚀 AZTRA LANGUAGE MODEL")
    logger.info("=" * 60)
    logger.info(f"🎯 Mod: {args.mode}")

    # Sistem kontrolü
    gpu_available, gpu_name, gpu_memory = check_gpu()

    # Gereksinimler kontrolü
    if args.mode != 'check':
        issues = check_requirements()
        if issues:
            logger.error("❌ Sistem kontrolü başarısız:")
            for issue in issues:
                logger.error(f"   - {issue}")
            return False

    try:
        # Modlar
        if args.mode == 'check':
            logger.info("🔍 Sistem kontrolü yapılıyor...")
            issues = check_requirements()

            if not issues:
                logger.info("✅ Tüm kontroller başarılı!")
                logger.info("💡 Şimdi tokenizer eğitimi yapabilirsiniz:")
                logger.info("   python main.py --mode train_tokenizer")
            else:
                logger.error("❌ Bazı sorunlar tespit edildi:")
                for issue in issues:
                    logger.error(f"   - {issue}")

        elif args.mode == 'train_tokenizer':
            logger.info("🔤 Tokenizer eğitimi başlıyor...")

            # Veri kontrolü
            if not os.path.exists(config['dataset_path_jsonl']) and not os.path.exists(config['dataset_path_txt']):
                logger.error(f"❌ Veri dosyası bulunamadı!")
                logger.error(f"   JSONL: {config['dataset_path_jsonl']}")
                logger.error(f"   TXT: {config['dataset_path_txt']}")
                logger.info("💡 Lütfen önce veri dosyanızı hazırlayın.")
                return False

            train_tokenizer_main()
            logger.info("✅ Tokenizer eğitimi tamamlandı!")
            logger.info("💡 Şimdi model eğitimi yapabilirsiniz:")
            logger.info("   python main.py --mode train")

        elif args.mode == 'train':
            logger.info("🚂 Model eğitimi başlıyor...")

            # Tokenizer kontrolü
            tokenizer_file = os.path.join(config['tokenizer_path'], 'tokenizer.json')
            if not os.path.exists(tokenizer_file):
                logger.error("❌ Tokenizer bulunamadı!")
                logger.info("💡 Önce tokenizer eğitimi yapın:")
                logger.info("   python main.py --mode train_tokenizer")
                return False

            # Veri kontrolü
            if not os.path.exists(config['dataset_path_txt']):
                logger.error(f"❌ Eğitim dosyası bulunamadı: {config['dataset_path_txt']}")
                return False

            # Eğitim süresi tahmini
            try:
                with open(config['dataset_path_txt'], 'r', encoding='utf-8') as f:
                    dataset_size = len(f.read().split())
                estimate_training_time(dataset_size, config)
            except Exception as e:
                logger.warning(f"⚠️ Eğitim süresi tahmini yapılamadı: {e}")

            trainModel()
            logger.info("✅ Model eğitimi tamamlandı!")
            logger.info("💡 Artık metin üretebilirsiniz:")
            logger.info("   python main.py --mode generate --prompt 'Merhaba'")

        elif args.mode == 'generate':
            if not args.prompt:
                logger.error("❌ Generate modu için --prompt parametresi gerekli")
                logger.info("💡 Örnek: python main.py --mode generate --prompt 'Merhaba dünya'")
                return False

            logger.info(f"🎨 Metin üretiliyor...")
            logger.info(f"📝 Prompt: '{args.prompt}'")

            result = generate_text(
                args.prompt,
                args.max_tokens,
                args.temperature,
                args.top_k,
                args.top_p
            )

            if result:
                print("\n" + "=" * 50)
                print("🎯 SONUÇ:")
                print("=" * 50)
                print(result)
                print("=" * 50)
            else:
                logger.error("❌ Metin üretilemedi")

        elif args.mode == 'interactive':
            logger.info("💬 İnteraktif mod başlıyor...")
            generate_interactive()

        return True

    except KeyboardInterrupt:
        logger.info("\n⏹️ Kullanıcı tarafından durduruldu")
        return False
    except Exception as e:
        logger.error(f"❌ Beklenmeyen hata: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


if __name__ == '__main__':
    #sys.argv = ['main.py', '--mode', 'check']
    #sys.argv = ['main.py', '--mode', 'train_tokenizer']
    #sys.argv = ['main.py', '--mode', 'train']
    #sys.argv = ['main.py', '--mode', 'generate', '--prompt', 'Python öğrenmek için nereden başlamalıyım?']
    sys.argv = ['main.py', '--mode', 'interactive']
    success = main()
    sys.exit(0 if success else 1)
