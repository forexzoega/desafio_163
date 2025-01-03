import os
import pyopencl as cl
import numpy as np
import random
from ecdsa import SECP256k1, SigningKey
import hashlib
from tqdm import tqdm
import time
import psutil
import base58
import logging
import yaml
import signal
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from typing import Optional
import json
import pickle
import argparse
import sys
import gc
from functools import lru_cache
import pywhatkit
import datetime

# Configuração de logging
logging.basicConfig(
    level=logging.DEBUG,  # Mude para DEBUG para mais detalhes
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('D:/bitcoin_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configurações padrão otimizadas para AMD
DEFAULT_CONFIG = {
    'gpu_memory_limit': 3621225472,       # 3GB (80% de 4GB)
    'ram_usage_limit_percentage': 90,      # 80% da RAM
    'gpu_batch_size': 30097152,            # 2M por lote (ajustado para RX 550)
    'checkpoint_interval': 10000,        # 1M tentativas
    'cleanup_interval': 3600,              # 1 hora
    'cpu_threads': multiprocessing.cpu_count()  # Usar todos os núcleos disponíveis
}

# Diretório para salvar checkpoints
CHECKPOINT_DIR = "D:/bitcoin_checkpoints"

# Kernel OpenCL otimizado para AMD GCN
OPENCL_KERNEL = """
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable

#define WORKGROUP_SIZE 128  // Ajustado para RX 550
#define VECTOR_SIZE 4

typedef struct {
    uchar4 data[16];  // 64 bytes alinhados para AMD GCN
} PrivateKey;

void generate_key_variation(__private uchar *key, ulong index, __local const uchar *template_local) {
    __constant uchar hex_chars[] = "0123456789abcdef";
    
    #pragma unroll 16
    for(int i = 0; i < 64; i += 4) {
        ((__private uchar4*)(key + i))[0] = ((__local uchar4*)(template_local + i))[0];
    }
    
    int pos = 0;
    #pragma unroll 8
    for(int i = 0; i < 64; i++) {
        if(template_local[i] == 'x') {
            int hex_val = (index >> (pos * 4)) & 0xF;
            key[i] = hex_chars[hex_val];
            pos++;
        }
    }
}

__kernel void bitcoin_key_search(
    __global const uchar* template_global,
    __global const uchar* target_address,
    __global volatile int* result_found,
    __global uchar* found_key,
    const ulong start_index,
    const ulong num_variations
) {
    ulong idx = get_global_id(0);
    ulong local_idx = get_local_id(0);
    
    __local uchar shared_template[64];
    
    // Copiar template global para local
    if(local_idx < 16) {
        ((__local uchar4*)(shared_template))[local_idx] = 
            ((__global uchar4*)(template_global))[local_idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (idx >= num_variations) return;
    
    __private uchar private_key[64];
    generate_key_variation(private_key, start_index + idx, shared_template);
    
    if (idx == 0) {
        atomic_xchg(result_found, 0);
        #pragma unroll 16
        for(int i = 0; i < 64; i += 4) {
            ((__global uchar4*)(found_key + i))[0] = 
                ((__private uchar4*)(private_key + i))[0];
        }
    }
}
"""

class GPUContext:
    def __init__(self):
        self.context = None
        self.queue = None
        self.program = None
        self.initialized = False
        self.build_options = [
            "-cl-std=CL1.2",              # AMD prefere CL1.2
            "-cl-mad-enable",
            "-cl-fast-relaxed-math",
            "-w",                         # Suprimir warnings
            f"-D WORKGROUP_SIZE={128}",   # Ajustado para RX 550
            "-D VECTOR_SIZE=4"            # AMD GCN suporta vetores de 4
        ]

    def initialize(self):
        if not self.initialized:
            try:
                platforms = cl.get_platforms()
                if not platforms:
                    raise RuntimeError("Nenhuma plataforma OpenCL encontrada")
                
                # Procurar especificamente por GPU AMD
                selected_device = None
                for platform in platforms:
                    if "AMD" in platform.name:
                        devices = platform.get_devices(device_type=cl.device_type.GPU)
                        if devices:
                            selected_device = devices[0]
                            break
                
                if not selected_device:
                    raise RuntimeError("GPU AMD não encontrada")

                logger.info(f"Usando GPU: {selected_device.name}")
                logger.info(f"Memória GPU: {selected_device.global_mem_size / (1024**3):.2f} GB")

                self.context = cl.Context([selected_device])
                self.queue = cl.CommandQueue(self.context)
                
                # Compilar programa com opções otimizadas para AMD
                self.program = cl.Program(self.context, OPENCL_KERNEL)
                try:
                    self.program.build(options=' '.join(self.build_options))
                except Exception as e:
                    build_log = self.program.get_build_info(selected_device, cl.program_build_info.LOG)
                    logger.error(f"Erro de compilação OpenCL:\n{build_log}")
                    raise
                
                self.initialized = True
                logger.info("GPU AMD inicializada com sucesso")
                
                # Log de uso de memória
                mem_info = selected_device.global_mem_size
                logger.info(f"Memória total da GPU: {mem_info / (1024**3):.2f} GB")
                logger.info(f"Memória compartilhada: {selected_device.local_mem_size / (1024**2):.2f} MB")
                
            except Exception as e:
                logger.error(f"Erro ao inicializar GPU: {str(e)}")
                raise

    def cleanup(self):
        if self.initialized:
            try:
                self.queue.finish()
            except:
                pass
            self.initialized = False
            logger.info("Recursos GPU liberados")

class BitcoinKeyFinder:
    def __init__(self, target_address: str, private_key_template: str, config_file: str = 'config.yaml'):
        self.target_address = target_address
        self.private_key_template = private_key_template.lower()
        self.num_x = self.private_key_template.count('x')
        self.total_combinations = 16 ** self.num_x
        self.config = self.load_config(config_file)
        self.tested_keys = set()
        self.running = True
        self.start_time = time.time()
        self.last_cleanup_time = time.time()
        self.last_checkpoint_time = time.time()
        self.last_stats_time = time.time()
        
        # Intervalos de tempo (em segundos)
        self.CLEANUP_INTERVAL = 300  # 5 minutos
        self.CHECKPOINT_INTERVAL = 600  # 10 minutos
        self.STATS_INTERVAL = 60  # 1 minuto
        
        self.setup_signal_handlers()
        self.gpu = GPUContext()
        
        # Carregar chaves testadas e ajustar batch size
        self.load_tested_keys()
        self.adjust_batch_size()
        
        logger.info(f"Iniciando busca com template: {self.private_key_template}")
        logger.info(f"Número de posições desconhecidas: {self.num_x}")
        logger.info(f"Total de combinações a testar: {self.total_combinations:,}")
        logger.info(f"Combinações restantes: {self.total_combinations - len(self.tested_keys):,}")
        
        self.print_stats()

    def load_config(self, config_file: str) -> dict:
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                if config is None:
                    config = {}
            return {**DEFAULT_CONFIG, **config}
        except FileNotFoundError:
            logger.warning(f"Arquivo de configuração {config_file} não encontrado. Usando configurações padrão.")
            return DEFAULT_CONFIG
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {e}")
            return DEFAULT_CONFIG

    def setup_signal_handlers(self):
        """Configurar handlers para sinais de interrupção"""
        def signal_handler(signum, frame):
            logger.info("Sinal de interrupção recebido. Salvando chaves testadas...")
            self.running = False
            self.save_tested_keys()  # Salvar chaves antes de encerrar
            logger.info("Chaves salvas. Encerrando...")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def save_checkpoint(self, attempts: int):
        checkpoint_data = {
            'attempts': attempts,
            'timestamp': time.time()
        }
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'checkpoint.json')
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # Criar diretório se não existir
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)
        logger.info(f"Checkpoint salvo: {attempts:,} tentativas")

    def load_checkpoint(self) -> int:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'checkpoint.json')
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            attempts = checkpoint_data.get('attempts', 0)
            logger.info(f"Checkpoint carregado: {attempts:,} tentativas")
            return attempts
        except FileNotFoundError:
            return 0
        except Exception as e:
            logger.error(f"Erro ao carregar checkpoint: {e}")
            return 0

    def load_tested_keys(self):
        """Carregar chaves testadas de um arquivo."""
        try:
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tested_keys.txt')
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    self.tested_keys = set(line.strip() for line in f)
                logger.info(f"Carregadas {len(self.tested_keys):,} chaves do arquivo")
            else:
                logger.info("Arquivo de chaves testadas não encontrado. Iniciando novo conjunto.")
                self.tested_keys = set()
        except Exception as e:
            logger.error(f"Erro ao carregar chaves testadas: {e}")
            self.tested_keys = set()

    def save_tested_keys(self):
        """Salvar chaves testadas em um arquivo de texto."""
        try:
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tested_keys.txt')
            # Garantir que o diretório existe
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Salvar as chaves
            with open(file_path, 'w') as f:
                for key in self.tested_keys:
                    f.write(f"{key}\n")
            
            logger.info(f"Chaves testadas salvas em '{file_path}' - Total: {len(self.tested_keys)}")
        except Exception as e:
            logger.error(f"Erro ao salvar chaves testadas: {e}")

    def cleanup_old_keys(self):
        current_time = time.time()
        old_keys = [k for k, timestamp in self.tested_keys.items() if current_time - timestamp > self.config['cleanup_interval']]
        for k in old_keys:
            del self.tested_keys[k]
        logger.info(f"Limpeza de memória: {len(old_keys)} chaves removidas")

    def process_batch_gpu(self, start_index: int, batch_size: int) -> Optional[str]:
        try:
            if not self.gpu.initialized:
                self.gpu.initialize()

            # Criar um array de dados na memória do sistema
            template_array = np.array([ord(c) for c in self.private_key_template], dtype=np.uint8)
            target_address_array = np.array([ord(c) for c in self.target_address], dtype=np.uint8)
            
            # Criar buffers na memória do sistema
            template_buffer = cl.Buffer(self.gpu.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                                     hostbuf=template_array)
            target_buffer = cl.Buffer(self.gpu.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                    hostbuf=target_address_array)
            result_found = np.zeros(1, dtype=np.int32)
            result_buffer = cl.Buffer(self.gpu.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                    hostbuf=result_found)
            found_key = np.zeros(64, dtype=np.uint8)
            found_key_buffer = cl.Buffer(self.gpu.context, cl.mem_flags.WRITE_ONLY,
                                       size=found_key.nbytes)

            # Executar kernel
            global_size = ((batch_size + 255) // 128) * 128  # Ajustado para o novo tamanho do workgroup
            logger.info(f"Executando kernel com global_size: {global_size}, batch_size: {batch_size}")
            self.gpu.program.bitcoin_key_search(
                self.gpu.queue, (global_size,), (128,),
                template_buffer, target_buffer, result_buffer, found_key_buffer,
                np.uint64(start_index), np.uint64(batch_size)
            )

            # Verificar resultado
            cl.enqueue_copy(self.gpu.queue, result_found, result_buffer)
            if result_found[0]:
                cl.enqueue_copy(self.gpu.queue, found_key, found_key_buffer)
                return ''.join(chr(x) for x in found_key if x != 0)

            return None

        except Exception as e:
            logger.error(f"Erro no processamento GPU: {e}")
            self.gpu.cleanup()
            return None

    def process_batch_hybrid(self, start_index: int, batch_size: int) -> Optional[str]:
        try:
            gpu_batch = int(batch_size * 0.7)  # 70% para GPU
            cpu_batch = batch_size - gpu_batch  # 30% para CPU
            
            # Processar lote na GPU
            gpu_result = self.process_batch_gpu(start_index, gpu_batch)
            if gpu_result:
                return gpu_result
            
            # Processar lote na CPU
            cpu_result = self.process_batch_cpu(
                start_index + gpu_batch,
                cpu_batch,
                self.private_key_template,
                self.target_address,
                self.tested_keys
            )
            
            if cpu_result:
                return cpu_result
            
            return None
            
        except Exception as e:
            self.handle_error(e, "processamento híbrido")
            return None

    def save_result(self, private_key: str):
        with open("chave_privada_encontrada.txt", "w") as f:
            f.write(f"Chave privada encontrada: {private_key}\n")
            f.write(f"Endereço Bitcoin: {self.target_address}\n")
            f.write(f"Data/Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        logger.info(f"Resultado salvo em 'chave_privada_encontrada.txt'")

    def send_whatsapp_notification(self, private_key: str):
        """Envia notificação via WhatsApp quando encontrar a chave"""
        try:
            phone_number = "+5521981496911"
            message = f"""
🔑 CHAVE PRIVADA ENCONTRADA! 🔑

Endereço Bitcoin: {self.target_address}
Chave Privada: {private_key}
Data/Hora: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
"""
            
            # Obtém hora e minuto atual
            now = datetime.datetime.now()
            hour = now.hour
            minute = now.minute + 1  # Agenda para o próximo minuto
            
            # Envia a mensagem
            pywhatkit.sendwhatmsg(phone_number, message, hour, minute)
            logger.info(f"Notificação WhatsApp enviada para {phone_number}")
        except Exception as e:
            logger.error(f"Erro ao enviar notificação WhatsApp: {e}")

    def run(self):
        try:
            logger.info("Iniciando busca por chave privada...")
            attempts = self.load_checkpoint()
            batch_size = self.config['gpu_batch_size']
            last_save = time.time()
            
            while attempts < self.total_combinations and self.running:
                try:
                    # Verificar se é hora de fazer limpeza de memória
                    current_time = time.time()
                    if current_time - self.last_cleanup_time >= self.CLEANUP_INTERVAL:
                        self.cleanup_memory()
                        self.last_cleanup_time = current_time
                    
                    # Verificar se é hora de fazer checkpoint
                    if current_time - self.last_checkpoint_time >= self.CHECKPOINT_INTERVAL:
                        self.auto_checkpoint()
                        self.last_checkpoint_time = current_time
                    
                    # Verificar se é hora de mostrar estatísticas
                    if current_time - self.last_stats_time >= self.STATS_INTERVAL:
                        self.print_stats()
                        self.last_stats_time = current_time
                    
                    # Ajustar batch size periodicamente
                    if attempts % 1000000 == 0:
                        self.adjust_batch_size()
                    
                    result = self.process_batch_hybrid(attempts, batch_size)
                    if result:
                        logger.info(f"Chave privada encontrada: {result}")
                        self.save_result(result)
                        self.save_tested_keys()
                        # Envia notificação WhatsApp
                        self.send_whatsapp_notification(result)
                        return True
                    
                    attempts += batch_size
                    
                    # Salvar progresso periodicamente
                    if current_time - last_save >= 300:  # A cada 5 minutos
                        self.save_tested_keys()
                        last_save = current_time
                    
                except Exception as e:
                    logger.error(f"Erro durante a execução: {e}")
                    self.handle_error(e, "execução principal")
                    time.sleep(1)
            
            self.save_tested_keys()
            logger.info("Busca finalizada")
            return False
            
        except Exception as e:
            logger.error(f"Erro fatal: {e}")
            self.handle_error(e, "execução principal")
            return False

    def process_batch_cpu(self, start_index: int, batch_size: int, template: str, target: str, tested_keys: set) -> Optional[str]:
        """Função para processamento em CPU"""
        try:
            hex_chars = "0123456789abcdef"
            
            for i in range(batch_size):
                idx = start_index + i
                
                # Gerar variação da chave privada
                private_key = list(template.lower())
                pos = 0
                for j in range(len(template)):
                    if template[j].lower() == 'x':
                        hex_val = random.randint(0, 15)
                        private_key[j] = hex_chars[hex_val]
                
                # Converter lista para string
                private_key_str = ''.join(private_key)
                
                # Verificar se a chave já foi testada
                if private_key_str in tested_keys:
                    continue
                
                # Adicionar a chave ao conjunto
                tested_keys.add(private_key_str)
                
                # Mostrar chave sendo testada em tempo real
                print(f"\rTestando chave: {private_key_str} | Total testadas: {len(tested_keys):,}", end='', flush=True)
                
                try:
                    # Verificar se a chave é válida
                    if not all(c in hex_chars for c in private_key_str):
                        continue
                    
                    # Gerar e verificar endereço Bitcoin
                    address = self.generate_bitcoin_address(private_key_str)
                    if address == target:
                        print("\nChave encontrada!")  # Nova linha quando encontrar
                        return private_key_str
                    
                except Exception as e:
                    continue
            
            return None
            
        except Exception as e:
            self.handle_error(e, "processamento CPU")
            return None

    @lru_cache(maxsize=1024)
    def generate_bitcoin_address(self, private_key_str: str) -> str:
        """Gera endereço Bitcoin a partir da chave privada com cache"""
        try:
            private_key_bytes = bytes.fromhex(private_key_str)
            signing_key = SigningKey.from_string(private_key_bytes, curve=SECP256k1)
            verifying_key = signing_key.get_verifying_key()
            public_key_bytes = verifying_key.to_string()
            sha256_hash = hashlib.sha256(public_key_bytes).digest()
            ripemd160_hash = hashlib.new('ripemd160', sha256_hash).digest()
            version_ripemd160_hash = b'\x00' + ripemd160_hash
            double_sha256 = hashlib.sha256(hashlib.sha256(version_ripemd160_hash).digest()).digest()
            binary_address = version_ripemd160_hash + double_sha256[:4]
            return base58.b58encode(binary_address).decode('ascii')
        except Exception as e:
            logger.error(f"Erro ao gerar endereço Bitcoin: {e}")
            return ""

    def cleanup_memory(self):
        """Limpa a memória periodicamente"""
        try:
            gc.collect()
            if hasattr(self.gpu, 'queue'):
                self.gpu.queue.finish()
            logger.info("Limpeza de memória realizada")
        except Exception as e:
            self.handle_error(e, "limpeza de memória")

    def auto_checkpoint(self):
        """Salva o estado atual automaticamente"""
        try:
            checkpoint_data = {
                'tested_keys_count': len(self.tested_keys),
                'timestamp': time.time(),
                'last_key': list(self.tested_keys)[-1] if self.tested_keys else None,
                'elapsed_time': time.time() - self.start_time
            }
            
            checkpoint_file = os.path.join(CHECKPOINT_DIR, 'checkpoint.json')
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f)
                
            logger.info(f"Checkpoint salvo: {len(self.tested_keys):,} chaves testadas")
        except Exception as e:
            self.handle_error(e, "salvamento de checkpoint")

    def print_stats(self):
        """Exibe estatísticas de performance com barra de progresso"""
        try:
            total_keys = len(self.tested_keys)
            remaining = self.total_combinations - total_keys
            progress = (total_keys / self.total_combinations) * 100
            elapsed_time = time.time() - self.start_time
            keys_per_second = total_keys / elapsed_time if elapsed_time > 0 else 0
            
            # Calcular tempo estimado restante
            if keys_per_second > 0:
                eta_seconds = remaining / keys_per_second
                eta_hours = eta_seconds / 3600
            else:
                eta_hours = float('inf')
            
            # Criar barra de progresso
            bar_length = 50
            filled_length = int(bar_length * progress / 100)
            bar = '=' * filled_length + '>' + '-' * (bar_length - filled_length - 1)
            
            # Obter informações de GPU
            gpu_info = ""
            if hasattr(self.gpu, 'context') and self.gpu.context:
                try:
                    devices = self.gpu.context.get_info(cl.context_info.DEVICES)
                    if devices:
                        gpu = devices[0]
                        gpu_info = f"""
- GPU: {gpu.name}
- Memória GPU: {gpu.global_mem_size / (1024**3):.2f} GB
- Uso GPU: {self.config['gpu_batch_size'] / gpu.global_mem_size * 100:.2f}%"""
                except:
                    gpu_info = "- GPU: Informação não disponível"
            
            stats = f"""
\n{'=' * 80}
Progresso da Busca Bitcoin:
[{bar}] {progress:.2f}%

Estatísticas:
- Chaves testadas: {total_keys:,} de {self.total_combinations:,}
- Chaves restantes: {remaining:,}
- Velocidade atual: {keys_per_second:.2f} chaves/s
- Tempo decorrido: {elapsed_time/3600:.2f} horas
- Tempo estimado restante: {eta_hours:.2f} horas
- Uso de RAM: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB
- Uso de CPU: {psutil.cpu_percent()}%{gpu_info}

Template: {self.private_key_template}
Endereço alvo: {self.target_address}
Última chave testada: {list(self.tested_keys)[-1] if self.tested_keys else "Nenhuma"}
{'=' * 80}
"""
            print(stats)
            logger.info("Estatísticas atualizadas")
        except Exception as e:
            self.handle_error(e, "exibição de estatísticas")

    def handle_error(self, error: Exception, context: str):
        """Tratamento centralizado de erros"""
        logger.error(f"Erro em {context}: {str(error)}")
        self.save_tested_keys()
        self.auto_checkpoint()
        
        if isinstance(error, (MemoryError, RuntimeError)):
            self.cleanup_memory()
            logger.info("Tentando recuperar após erro de memória")
        
        return None

    def adjust_batch_size(self):
        """Ajusta o tamanho do lote baseado na performance"""
        try:
            available_memory = psutil.virtual_memory().available
            gpu_memory = getattr(self.gpu, 'global_mem_size', None)
            
            # Ajusta batch_size baseado na memória disponível
            new_batch_size = min(
                self.config['gpu_batch_size'],
                int(available_memory * 0.8 / 256),  # 256 bytes por chave
                int(gpu_memory * 0.8 / 256) if gpu_memory else float('inf')
            )
            
            self.config['gpu_batch_size'] = new_batch_size
            logger.info(f"Batch size ajustado para: {new_batch_size:,}")
        except Exception as e:
            self.handle_error(e, "ajuste de batch size")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Buscar chaves privadas Bitcoin.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Arquivo de configuração')
    parser.add_argument('--template', type=str, required=True, help='Template da chave privada')
    parser.add_argument('--target', type=str, required=True, help='Endereço Bitcoin alvo')
    return parser.parse_args()

def main():
 # Converter o template para minúsculas para garantir consistência
    private_key_template = "403b3d4fcff56a92f335a0cf570e4xbxb17b2a6x867x86a84x0x8x3x3x3x7x3x"
    target_address = "1Hoyt6UBzwL5vvUSTLMQC2mwvvE5PpeSC"
    finder = None

    # Validar o template antes de começar
    valid_chars = set("0123456789abcdefx")
    if not all(c in valid_chars for c in private_key_template):
        logger.error("Template contém caracteres inválidos")
        return 1

    try:
        finder = BitcoinKeyFinder(target_address, private_key_template)
        finder.run()
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        return 1
    finally:
        if finder and hasattr(finder, 'cpu_pool'):
            finder.cpu_pool.shutdown()
        if finder:  # Salvar chaves testadas ao final
            finder.save_tested_keys()
    return 0

if __name__ == "__main__":
    exit(main())