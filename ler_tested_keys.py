import os

# Caminho do arquivo
file_path = 'tested_keys.txt'

try:
    # Verificar se o arquivo existe
    if not os.path.exists(file_path):
        print(f"O arquivo {file_path} não foi encontrado.")
    else:
        # Abrir o arquivo e contar as chaves testadas
        with open(file_path, 'r') as f:
            tested_keys = set(line.strip() for line in f)
        
        # Exibir apenas o total de chaves testadas
        print(f"Total de chaves testadas: {len(tested_keys):,}")

except Exception as e:
    print(f"Ocorreu um erro ao abrir o arquivo: {e}")