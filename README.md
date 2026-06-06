# Projeto A3 - Computação Gráfica | Processamento Digital de Imagens Ambientais

Este repositório contém a implementação do Projeto Prático (Avaliação A3) da disciplina de Computação Gráfica e Realidade Virtual. O objetivo do código é processar uma imagem de satélite (capturada via Earth Explorer) para identificar, isolar e calcular a área afetada por uma queimada utilizando técnicas de Visão Computacional.

## Tecnologias e Bibliotecas Utilizadas
- **Linguagem:** Python 3.12+
- **OpenCV (`cv2`):** Leitura, conversão de cores, filtros (Gaussiano, Mediana, Bilateral), detecção de bordas (Canny) e contornos.
- **scikit-image:** Limiarização automática de Otsu para segmentação.
- **NumPy:** Manipulação de matrizes e cálculo de métricas.
- **Matplotlib:** Geração de gráficos (histogramas) e grids comparativos.

## Estrutura de Arquivos
- `projeto_a3_ryan.py`: Código-fonte principal com o pipeline de processamento.
- `imagem_original.png`: Imagem de satélite base utilizada na análise.
- `resultados/`: Diretório gerado automaticamente contendo as imagens isoladas, o histograma e o grid comparativo final.
- `Artigo_a3.pdf`: Relatório técnico com a fundamentação e análise crítica dos resultados.

## Como Executar o Projeto

1. **Clone o repositório ou extraia os arquivos:**

   No terminal clone o repositório, navegue até a pasta do projeto e acesse pela IDE (ex: code .).
   ```bash
   git clone https://github.com/ryanpaiva-coder/projeto-a3-cg.git
   cd pasta/projeto_a3_cg/
   code .
   ```
   Substitua "pasta" pela pasta na qual o projeto foi salvo e o "code ." pela sua IDE.
   

2. **Crie e ative um ambiente virtual (Recomendado) e execute o projeto:**
   No Linux/macOS:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   No Windows:
   ```DOS
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Instale as dependências necessárias:**
   ```bash
   pip install opencv-python numpy matplotlib pillow scikit-image
   ```

4. **Execute o script principal:**
   ```bash
   python3 projeto_a3_ryan.py
   ```

5. **Verifique os resultados:**

   Após a execução, o terminal exibirá as métricas calculadas (PSNR, SNR e Área Segmentada e Valor Médio de Pixel) e todas as imagens processadas estarão salvas dentro da pasta `resultados/`.
