import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology
import os

# Lê e converte BGR para RGB, exibe metadados
def carregar_imagem(path):
    # Lê a imagem em BGR
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print(f"Erro: Não foi possível carregar a imagem em {path}")
        return None

    # Converte BGR para RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Pega e exibe os metadados exigidos 
    altura, largura, canais = img_rgb.shape
    tamanho_kb = os.path.getsize(path) / 1024

    print("--- Metadados da Imagem Original ---")
    print(f"Resolução: {largura}x{altura} pixels")
    print(f"Número de canais: {canais}")
    print(f"Tamanho em disco: {tamanho_kb:.2f} KB\n")

    return img_rgb

# Escala de cinza, resize, normalização
def pre_processar(img):
    # Converte para escala de cinza
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Redimensionamento para largura de 800px
    largura_nova = 800
    proporcao = largura_nova / float(img.shape[1])
    altura_nova = int(float(img.shape[0]) * proporcao)
    img_resized = cv2.resize(img_gray, (largura_nova, altura_nova))

    # Normalização [0,1] - convertendo de uint8 (0-255) para float32
    img_norm = img_resized.astype(np.float32) / 255.0

    return img_gray, img_resized, img_norm

# Plota distribuição de intensidades por canal
def analisar_histograma(img):
    cores = ('r', 'g', 'b')
    plt.figure(figsize=(10, 5))
    plt.title('Histograma dos Canais R, G e B da Imagem Original')
    plt.xlabel('Intensidade do Pixel (0-255)')
    plt.ylabel('Frequência')

    for i, cor in enumerate(cores):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=cor)
        plt.xlim([0, 256])

    plt.grid(alpha=0.3)
    
    # Salva a imagem na pasta resultados (ignorar)
    plt.savefig('resultados/histograma_original.png')
    print("Histograma salvo em 'resultados/histograma_original.png'")
    plt.close()

# Gaussiano, Mediana e Bilateral
def aplicar_filtros(img):
    # Filtro Gaussiano com sigma = 1.5
    # O tamanho do kernel (0,0) para o OpenCV calcular automaticamente com base no sigma
    img_gaussian = cv2.GaussianBlur(img, (0, 0), 1.5)
    
    # Filtro de Mediana com kernel 5x5
    img_median = cv2.medianBlur(img, 5)
    
    # Filtro Bilateral
    # d=9 (diâmetro da vizinhança), sigmaColor=75, sigmaSpace=75 (parâmetros padrões recomendados)
    img_bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    
    return img_gaussian, img_median, img_bilateral

# Canny com limiares adaptativos
def detectar_bordas(img_gray):
    # Operador Canny com limiares de 50 e 150
    bordas = cv2.Canny(img_gray, 50, 150)
    return bordas

# Otsu + erosão/dilatação + detecção de contornos
def segmentar(img_gray):
    # Limiarização de Otsu
    limiar = filters.threshold_otsu(img_gray)
    mascara_otsu = img_gray > limiar
    
    # Converte booleano para uint8, que é necessário para o OpenCV
    mascara_uint8 = (mascara_otsu * 255).astype(np.uint8)
    
    # Morfologia matemática: Erosão seguida de dilatação com kernel 3x3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_erodida = cv2.erode(mascara_uint8, kernel, iterations=1)
    mascara_refinada = cv2.dilate(img_erodida, kernel, iterations=1)
    
    # Detecção de contornos na máscara final
    contornos, _ = cv2.findContours(mascara_refinada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return mascara_uint8, mascara_refinada, contornos

# PSNR, SNR e percentual de área segmentada
def calcular_metricas(orig, proc, mascara):
    # PSNR e SNR avaliam a qualidade do filtro, comparando a cinza original com a suavizada
    psnr = cv2.PSNR(orig, proc)
    
    # Cálculo aproximado de SNR: Média do sinal dividida pelo desvio padrão do ruído
    ruido = cv2.absdiff(orig, proc)
    snr = np.mean(orig) / (np.std(ruido) + 1e-10) # 1e-10 para evitar divisão por zero
    
    # Percentual de área segmentada da queimada
    area_total = mascara.shape[0] * mascara.shape[1]
    area_segmentada = np.count_nonzero(mascara)
    percentual_area = (area_segmentada / area_total) * 100

    #Valor médio de pixel da região segmentada
    valor_medio_pixel = cv2.mean(orig, mask=mascara)[0]
    
    return psnr, snr, percentual_area, valor_medio_pixel

# Grid visual comparativo
def plotar_resultados(imgs, titulos):
    n_imgs = len(imgs)
    linhas = (n_imgs + 2) // 3 
    
    plt.figure(figsize=(15, 5 * linhas))
    for i in range(n_imgs):
        plt.subplot(linhas, 3, i + 1)
        # Se for imagem de 1 canal, cinza ou binária, precisa do cmap='gray'
        if len(imgs[i].shape) == 2:
            plt.imshow(imgs[i], cmap='gray')
        else:
            plt.imshow(imgs[i])
        plt.title(titulos[i])
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig('resultados/grid_comparativo.png')
    print("Grid salvo em 'resultados/grid_comparativo.png'")
    plt.close()

# Salva cada imagem do processo isoladamente
def salvar_imagens_isoladas(imgs, titulos):
    print("\nSalvando imagens isoladas...")
    for img, titulo in zip(imgs, titulos):
        # Formata o título para virar um nome de arquivo válido (ignorar)
        nome_arquivo = titulo.replace(" ", "_").replace("(", "").replace(")", "").lower()
        caminho = f"resultados/{nome_arquivo}.png"
        
        # Se a imagem tiver 3 canais, converte para BGR para salvar corretamente
        if len(img.shape) == 3:
            img_salvar = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            # Se for escala de cinza ou máscara binária, não precisa converter
            img_salvar = img
            
        cv2.imwrite(caminho, img_salvar)
        print(f" - Salva: {caminho}")
    
def explicar_matematica_filtros():
        print("\n" + "="*60)
        print("   FUNDAMENTAÇÃO MATEMÁTICA DOS FILTROS APLICADOS")
        print("="*60)
        
        print("1. Filtro Gaussiano:")
        print("   - Calculo: Convolução usando uma distribuição normal.")
        print("   - Formula 2D: G(x,y) = (1 / 2*pi*sigma^2) * exp(-(x^2 + y^2) / 2*sigma^2)")
        print("   - Efeito: Media ponderada onde o centro tem mais peso. Suaviza ruídos globais.\n")
        
        print("2. Filtro de Mediana (Não-linear):")
        print("   - Calculo: Ordena os valores dos pixels no kernel (ex: 5x5) e pega o valor central.")
        print("   - Efeito: Excelente para remover ruídos do tipo 'sal e pimenta' preservando bordas.\n")
        
        print("3. Filtro Bilateral:")
        print("   - Calculo: Multiplica dois kernels Gaussianos (um de distância espacial e um de intensidade).")
        print("   - Efeito: Suaviza a imagem, mas 'desliga' a suavização onde há uma mudança brusca")
        print("     de cor. É o melhor para preservar bordas nítidas.")
        print("="*60 + "\n")

# Aqui ocorre a chamada das funções uma a uma e passo a passo
if __name__ == "__main__":
    print("Iniciando processamento de imagens ambientais...")
    caminho_imagem = "imagem_original.png"
    
    # 1. Carregar a imagem
    img_original = carregar_imagem(caminho_imagem)
    
    if img_original is not None:
        # 2. Exibir o Histograma
        analisar_histograma(img_original)
        
        # 3. Pré-processar
        img_gray, img_resized, img_norm = pre_processar(img_original)
        print("Pré-processamento concluído (Escala de cinza, Redimensionamento, Normalização).")
    
        # 4. Aplicar Filtros
        img_gaussian, img_median, img_bilateral = aplicar_filtros(img_resized)
        print("Filtros Gaussiano, Mediana e Bilateral aplicados com sucesso.")

        # Chama a explicação matemática aqui (ignorar)
        explicar_matematica_filtros()

        # 5. Detectar Bordas
        bordas_canny = detectar_bordas(img_resized)
        print("Detecção de bordas (Canny) concluída.")
    
        # 6. Segmentação
        mascara_otsu, mascara_refinada, contornos = segmentar(img_resized)
        print(f"Segmentação concluída. Encontrados {len(contornos)} contornos.")

        # 7. Métricas
        # Usamos o filtro Gaussiano como parâmetro de imagem processada para as métricas
        psnr, snr, perc_area, valor_medio = calcular_metricas(img_resized, img_gaussian, mascara_refinada)
        print(f"\n--- Métricas de Avaliação ---")
        print(f"PSNR (Original vs Gaussiano): {psnr:.2f} dB")
        print(f"SNR: {snr:.2f}")
        print(f"Área Segmentada: {perc_area:.2f}%")
        print(f"Valor Médio de Pixel (Área Segmentada): {valor_medio:.2f}\n")
    
        # 8. Preparar imagem final com contornos sobrepostos
        # Redimensiona a colorida original para o mesmo tamanho das máscaras
        largura_nova = 800
        proporcao = largura_nova / float(img_original.shape[1])
        altura_nova = int(float(img_original.shape[0]) * proporcao)
        img_contornos = cv2.resize(img_original, (largura_nova, altura_nova))
    
        # Desenha os contornos em vermelho (RGB: 255, 0, 0) com espessura 2
        cv2.drawContours(img_contornos, contornos, -1, (255, 0, 0), 2)
    
        # 9. Plotar Resultados no Grid
        imagens_plot = [
            cv2.resize(img_original, (largura_nova, altura_nova)), img_resized, img_gaussian,
            img_median, img_bilateral, bordas_canny,
            mascara_otsu, mascara_refinada, img_contornos
        ]
        titulos_plot = [
            "Original (Redimensionada)", "Escala de Cinza", "Filtro Gaussiano",
            "Filtro Mediana", "Filtro Bilateral", "Bordas (Canny)",
            "Máscara Otsu", "Máscara Refinada", "Contornos Destacados"
        ]
        
        plotar_resultados(imagens_plot, titulos_plot)

        # 10. Salvar imagens isoladas para análise isolada
        salvar_imagens_isoladas(imagens_plot, titulos_plot)

        print("Processamento finalizado com sucesso! Confira a pasta 'resultados/'.")