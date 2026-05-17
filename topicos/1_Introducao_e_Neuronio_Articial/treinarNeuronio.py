#!/usr/bin/env python3
"""Script de treinamento do neuronio artificial - Topico 1 da disciplina."""

import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os

from neuronio import ArtificialNeuron

def baixar_dataset(url, caminho_destino):
    """Baixa o dataset se ele nao existir."""
    if not os.path.exists(caminho_destino):
        print(f"Baixando dataset de {url}...")
        urllib.request.urlretrieve(url, caminho_destino)
        print("Download completo!")
        return True
    else:
        print(f"Dataset ja existe em {caminho_destino}")
        return False

def main():
    """Executa o treinamento completo do neuronio artificial."""
    
    print("=" * 70)
    print("Topico 1 - Introducao e Neuronio Artificial")
    print("=" * 70)
    
    # Configuracao
    url_dataset = "http://inf.ufrgs.br/~artavares/datasets/alegrete.csv"
    caminho_dataset = "data/alegrete.csv"
    
    # Hiperparametros (voce pode modificar aqui)
    TAXA_APRENDIZADO = 0.01
    NUM_EPOCAS = 100
    B_INICIAL = 0
    W_INICIAL = 0
    
    # Criar diretorios necessarios
    os.makedirs("data", exist_ok=True)
    os.makedirs("resultados", exist_ok=True)
    
    # 1. Carregar dados
    print("\n1. Carregando dataset...")
    baixar_dataset(url_dataset, caminho_dataset)
    
    dataset = np.genfromtxt(caminho_dataset, delimiter=',')
    X = dataset[:, 0]  # Area (hectares)
    Y = dataset[:, 1]  # Preco (milhares de R$)
    
    print(f"   Dataset carregado: {len(X)} amostras")
    print(f"   Area: min={X.min():.2f}, max={X.max():.2f}, media={X.mean():.2f}")
    print(f"   Preco: min={Y.min():.2f}, max={Y.max():.2f}, media={Y.mean():.2f}")
    
    # 2. Criar e treinar modelo
    print("\n2. Treinando neuronio artificial...")
    print(f"   Hiperparametros: alpha={TAXA_APRENDIZADO}, epocas={NUM_EPOCAS}")
    print(f"   Valores iniciais: b={B_INICIAL}, w={W_INICIAL}")
    
    model = ArtificialNeuron(alpha=TAXA_APRENDIZADO)
    b_history, w_history = model.fit(X, Y, num_epochs=NUM_EPOCAS, 
                                      initial_b=B_INICIAL, initial_w=W_INICIAL)
    
    # 3. Resultados
    print("\n3. Resultados do treinamento:")
    mse_final = model.compute_mse(X, Y)
    print(f"   MSE final: {mse_final:.4f}")
    print(f"   Curva encontrada: preco = {w_history[-1]:.4f} * area + {b_history[-1]:.4f}")
    
    # Calcular MSE inicial corretamente
    model_temp = ArtificialNeuron(alpha=TAXA_APRENDIZADO)
    _, _ = model_temp.fit(X, Y, num_epochs=0, initial_b=B_INICIAL, initial_w=W_INICIAL)
    mse_inicial = model_temp.compute_mse(X, Y)
    
    reducao_percentual = ((mse_inicial - mse_final) / mse_inicial) * 100
    print(f"   Reducao do erro: {reducao_percentual:.2f}%")
    
    # 4. Gerar graficos
    print("\n4. Gerando visualizacoes...")
    
    # Calcular MSE ao longo do treinamento
    mse_list = []
    for i in range(len(b_history)):
        model_temp = ArtificialNeuron(alpha=TAXA_APRENDIZADO)
        model_temp.b = b_history[i]
        model_temp.w = w_history[i]
        mse_list.append(model_temp.compute_mse(X, Y))
    
    # Criar figura com 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Topico 1 - Neuronio Artificial: Regressao Linear\n'
                 f'alpha={TAXA_APRENDIZADO}, epocas={NUM_EPOCAS}, MSE final={mse_final:.4f}', 
                 fontsize=12, fontweight='bold')
    
    # Grafico 1: Evolucao do MSE
    axes[0].plot(mse_list, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoca (iteracao)')
    axes[0].set_ylabel('Erro Quadratico Medio (MSE)')
    axes[0].set_title('Evolucao do Erro durante o Treinamento')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    axes[0].axhline(y=mse_final, color='r', linestyle='--', alpha=0.5, 
                    label=f'MSE final: {mse_final:.4f}')
    axes[0].legend()
    
    # Grafico 2: Dados reais vs. Predicao final
    axes[1].scatter(X, Y, alpha=0.6, s=30, label='Dados reais', color='blue')
    predictions = w_history[-1] * X + b_history[-1]
    idx_ordenado = np.argsort(X)
    axes[1].plot(X[idx_ordenado], predictions[idx_ordenado], 'r-', linewidth=2, 
                 label=f'Predicao: {w_history[-1]:.2f}*x + {b_history[-1]:.2f}')
    axes[1].set_xlabel('Area (hectares)')
    axes[1].set_ylabel('Preco (milhares de R$)')
    axes[1].set_title('Resultado Final: Dados vs. Regressao')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Grafico 3: Trajetoria do gradiente
    axes[2].plot(b_history, w_history, 'g-', alpha=0.7, linewidth=1.5)
    axes[2].scatter(b_history[0], w_history[0], c='green', s=100, 
                    label=f'Inicio (b={b_history[0]:.2f}, w={w_history[0]:.2f})', 
                    marker='o', edgecolors='black', zorder=5)
    axes[2].scatter(b_history[-1], w_history[-1], c='red', s=100, 
                    label=f'Fim (b={b_history[-1]:.2f}, w={w_history[-1]:.2f})', 
                    marker='s', edgecolors='black', zorder=5)
    axes[2].set_xlabel('Bias (b)')
    axes[2].set_ylabel('Peso (w)')
    axes[2].set_title('Trajetoria dos Parametros no Gradiente Descendente')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar figura
    output_path = 'resultados/graficos_topico1.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Grafico salvo: {output_path}")
    
    # Mostrar graficos
    if not os.environ.get('TESTING'):
        plt.show()
    else:
        plt.close()
    
    # 5. Salvar resultados em arquivo texto
    print("\n5. Salvando resultados...")
    with open('resultados/parametros_finais.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Topico 1 - Neuronio Artificial\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Hiperparametros:\n")
        f.write(f"  Taxa de aprendizado (alpha): {TAXA_APRENDIZADO}\n")
        f.write(f"  Numero de epocas: {NUM_EPOCAS}\n")
        f.write(f"  Parametros iniciais: b={B_INICIAL}, w={W_INICIAL}\n\n")
        f.write(f"Resultados finais:\n")
        f.write(f"  Bias final (b): {b_history[-1]:.6f}\n")
        f.write(f"  Peso final (w): {w_history[-1]:.6f}\n")
        f.write(f"  MSE final: {mse_final:.6f}\n")
        f.write(f"  Reducao do erro: {reducao_percentual:.2f}%\n\n")
        f.write(f"Equacao da reta:\n")
        f.write(f"  preco = {w_history[-1]:.6f} * area + {b_history[-1]:.6f}\n")
    
    print(f"   Resultados salvos em: resultados/parametros_finais.txt")
    
    # 6. Resumo final
    print("\n" + "=" * 70)
    print("Topico 1 concluido com sucesso!")
    print("=" * 70)
    print("Graficos e resultados salvos em: topicos/1_Introducao_e_Neuronio_Articial/resultados/")
    print("\nDica: Voce pode modificar os hiperparametros editando este arquivo")
    print("  (TAXA_APRENDIZADO, NUM_EPOCAS, B_INICIAL, W_INICIAL)\n")

if __name__ == "__main__":
    main()