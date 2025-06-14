"""
MÓDULO PRINCIPAL: SISTEMA DE GERAÇÃO DE JOGOS DE LOTERIA AVANÇADO

Este sistema implementa um gerador inteligente de jogos de loteria que combina:
- Análise estatística de resultados históricos
- Modelos de machine learning (ensemble)
- Validação baseada em regras probabilísticas
- Otimização geométrica no volante

Autor: Natanael
Versão: 1.0.0
Data: 14/06/2025
"""

import numpy as np
import pandas as pd
from collections import Counter
import random
import math
import os
import tkinter as tk
from tkinter import filedialog
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional, Set
import logging
from sklearn.utils import resample
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
import sys
import subprocess
from scipy.spatial.distance import pdist, squareform

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LotteryConfig:
    """
    Configurações globais do sistema de geração de jogos.

    Atributos:
        MIN_EVEN (int): Mínimo de números pares em um jogo válido (padrão: 4)
        MAX_EVEN (int): Máximo de números pares em um jogo válido (padrão: 11)
        MIN_ODD (int): Mínimo de números ímpares em um jogo válido (padrão: 4)
        MAX_ODD (int): Máximo de números ímpares em um jogo válido (padrão: 11)
        MIN_LOW (int): Mínimo de números baixos (1-12) em um jogo válido (padrão: 4)
        MAX_LOW (int): Máximo de números baixos (1-12) em um jogo válido (padrão: 11)
        MIN_SUM (int): Soma mínima dos números em um jogo válido (padrão: 100)
        MAX_SUM (int): Soma máxima dos números em um jogo válido (padrão: 250)
        FREQ_WEIGHT (float): Peso da frequência histórica no score (0.0-1.0, padrão: 0.40)
        DELAY_WEIGHT (float): Peso do atraso do número no score (0.0-1.0, padrão: 0.35)
        PROB_WEIGHT (float): Peso da probabilidade do modelo no score (0.0-1.0, padrão: 0.25)
        NUMBERS_PER_GAME (int): Quantidade de números em cada jogo (padrão: 15)
        TOTAL_NUMBERS (int): Faixa total de números da loteria (padrão: 25)
        GAMES_TO_GENERATE (int): Quantidade padrão de jogos a gerar (padrão: 7)
    """
    # Parâmetros de validação
    MIN_EVEN = 4
    MAX_EVEN = 11
    MIN_ODD = 4
    MAX_ODD = 11
    MIN_LOW = 4
    MAX_LOW = 11
    MIN_SUM = 100
    MAX_SUM = 250

    # Pesos e parâmetros de probabilidade
    FREQ_WEIGHT = 0.40
    DELAY_WEIGHT = 0.35
    PROB_WEIGHT = 0.25
    TREND_WEIGHT = 0.10
    GEOMETRY_WEIGHT = 0.15

    # Configurações de atraso
    LOW_DELAY_THRESHOLD = 3
    MEDIUM_DELAY_THRESHOLD = 7
    MAX_DELAY = 12
    LOW_DELAY_WEIGHT = 1.0
    MEDIUM_DELAY_WEIGHT = 0.7
    HIGH_DELAY_WEIGHT = 0.3

    # Configurações do jogo
    NUMBERS_PER_GAME = 15
    TOTAL_NUMBERS = 25
    GAMES_TO_GENERATE = 7
    NEGATIVE_SAMPLES_MULTIPLIER = 5
    TREND_WINDOW_SIZE = 10


class EnhancedLotteryConfig(LotteryConfig):
    """
    Configurações estendidas com novos parâmetros avançados.
    Herda todos os atributos de LotteryConfig e adiciona:

    Atributos Adicionais:
        CONSECUTIVE_PAIR_WEIGHT (float): Peso para pares consecutivos frequentes
        CORRELATION_PENALTY (float): Penalidade para números altamente correlacionados
        QUADRANT_MIN (int): Mínimo de números por quadrante no volante
        MODEL_WEIGHTS (List[float]): Pesos para os modelos no ensemble [MLP, RF, XGBoost]
        SEASONAL_WINDOWS (List[int]): Janelas temporais para análise sazonal
    """
    REQUIRED_ADVANCED_CHECKS = 1
    QUADRANT_MIN = 2
    MAX_GENERATION_ATTEMPTS = 100
    REQUIRED_ADVANCED_CHECKS = 2

    # Novos parâmetros para análise avançada
    CONSECUTIVE_PAIR_WEIGHT = 0.15
    CORRELATION_PENALTY = 0.2
    QUADRANT_MIN = 2
    SA_TEMPERATURE = 1.0
    SA_COOLING_RATE = 0.95
    SA_ITERATIONS = 100

    # Parâmetros do ensemble
    MODEL_WEIGHTS = [0.4, 0.3, 0.3]  # Pesos para MLP, RF, XGBoost

    # Configurações de análise temporal
    SEASONAL_WINDOWS = [10, 30, 100]  # Janelas para análise sazonal
    TREND_SENSITIVITY = 0.25


class LotteryWheel:
    """
    Representação geométrica do volante da loteria.

    Atributos:
        POSITIONS (Dict[int, Tuple[int, int]]): Mapeamento de números para posições (linha, coluna)
        COMMON_PATTERNS (List[Set[int]]): Padrões comuns a serem evitados nos jogos

    Exemplo:
       # >>> LotteryWheel.POSITIONS[1]
        (0, 0)  # Número 1 está na linha 0, coluna 0
    """
    POSITIONS = {
        1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (0, 3), 5: (0, 4),
        6: (1, 0), 7: (1, 1), 8: (1, 2), 9: (1, 3), 10: (1, 4),
        11: (2, 0), 12: (2, 1), 13: (2, 2), 14: (2, 3), 15: (2, 4),
        16: (3, 0), 17: (3, 1), 18: (3, 2), 19: (3, 3), 20: (3, 4),
        21: (4, 0), 22: (4, 1), 23: (4, 2), 24: (4, 3), 25: (4, 4)
    }

    COMMON_PATTERNS = [
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},  # Primeiras 15 dezenas
        {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},  # Últimas 15 dezenas
        {5, 10, 15, 20, 25, 1, 6, 11, 16, 21, 2, 7, 12, 17, 22}  # Padrão em cruz
    ]


class AdvancedLotteryWheel(LotteryWheel):
    """
    Representação geométrica estendida do volante com análise por quadrantes.
    Herda de LotteryWheel e adiciona funcionalidades de quadrantes.

    Atributos:
        QUADRANTS (Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]]):
            Definição dos quadrantes (Q1 a Q4) com limites de linhas e colunas

    Métodos:
        get_quadrant(number): Retorna o quadrante (1-4) de um número específico
    """
    QUADRANTS = {
        1: [(0, 2), (0, 2)],  # Q1: linhas 0-2, cols 0-2
        2: [(0, 2), (3, 4)],  # Q2: linhas 0-2, cols 3-4
        3: [(3, 4), (0, 2)],  # Q3: linhas 3-4, cols 0-2
        4: [(3, 4), (3, 4)]  # Q4: linhas 3-4, cols 3-4
    }

    @classmethod
    def get_quadrant(cls, number: int) -> int:
        """
        Retorna o quadrante (1-4) de um número específico no volante.

        Parâmetros:
            number (int): O número a ser analisado (1-25)

        Retorna:
            int: Número do quadrante (1-4) ou 0 se fora dos quadrantes definidos

        Exemplo:
          #  >>> AdvancedLotteryWheel.get_quadrant(1)
            1  # Quadrante 1
          #  >>> AdvancedLotteryWheel.get_quadrant(25)
            4  # Quadrante 4
        """
        row, col = cls.POSITIONS[number]
        for quad, ((r_min, r_max), (c_min, c_max)) in cls.QUADRANTS.items():
            if r_min <= row <= r_max and c_min <= col <= c_max:
                return quad
        return 0


class FileHandler:
    """
    Classe para manipulação de arquivos de resultados históricos.
    Suporta formatos Excel (.xlsx, .xls) e texto (.txt).

    Métodos Principais:
        select_file(): Abre diálogo para seleção de arquivo
        read_results(file_path): Lê resultados do arquivo especificado
        create_example_file(file_path): Cria arquivo de exemplo com dados fictícios
    """

    @staticmethod
    def select_file() -> Optional[str]:
        """
        Abre diálogo gráfico para seleção do arquivo de resultados.

        Retorna:
            Optional[str]: Caminho do arquivo selecionado ou None se cancelado

        Exemplo:
        #    >>> file_path = FileHandler.select_file()
        #    >>> print(f"Arquivo selecionado: {file_path}")
        """
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            file = filedialog.askopenfilename(
                title="Selecione o arquivo de resultados",
                filetypes=[
                    ("Excel files", "*.xlsx *.xls"),
                    ("Text files", "*.txt"),
                    ("All files", "*.*")
                ],
                initialdir=os.getcwd()
            )
            root.destroy()
            return file if file else None
        except Exception as e:
            logger.error(f"Erro ao selecionar arquivo: {e}")
            return None

    @staticmethod
    def read_results(file_path: str) -> List[List[int]]:
        """
        Lê resultados históricos de arquivo Excel ou texto.

        Parâmetros:
            file_path (str): Caminho completo do arquivo

        Retorna:
            List[List[int]]: Lista de jogos, cada jogo é uma lista de inteiros

        Levanta:
            Exception: Se o arquivo não existe ou formato é inválido

        Exemplo:
        #    >>> resultados = FileHandler.read_results("historico.xlsx")
        #    >>> print(f"Total de jogos carregados: {len(resultados)}")
        """
        if not os.path.exists(file_path):
            logger.error(f"Arquivo não encontrado: {file_path}")
            return []

        try:
            # Para arquivos Excel
            if file_path.lower().endswith(('.xlsx', '.xls')):
                if not install_excel_deps():
                    raise ImportError("Dependências para Excel não disponíveis")

                try:
                    df = pd.read_excel(
                        file_path,
                        header=None,
                        engine='openpyxl'
                    )
                    return [
                        row.dropna().astype(int).tolist()
                        for _, row in df.iterrows()
                        if not row.dropna().empty
                    ]
                except Exception as e:
                    logger.error(f"Erro ao ler arquivo Excel: {e}")
                    return []

            # Para arquivos texto
            elif file_path.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return [
                        list(map(int, line.strip().replace(',', ' ').split()))
                        for line in f if line.strip()
                    ]

            else:
                logger.error(f"Formato de arquivo não suportado: {file_path}")
                return []

        except Exception as e:
            logger.error(f"Erro ao processar arquivo: {e}")
            return []

    @staticmethod
    def create_example_file(file_path: str):
        """
        Cria um arquivo de exemplo com dados fictícios no formato especificado.

        Parâmetros:
            file_path (str): Caminho do arquivo a ser criado (determina o formato)

        Exemplo:
         #   >>> FileHandler.create_example_file("exemplo.xlsx")
            # Cria arquivo Excel com 2 jogos de exemplo
        """
        try:
            import pandas as pd
            example_data = [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 1, 3, 5]
            ]
            df = pd.DataFrame(example_data)

            if file_path.endswith('.xlsx'):
                df.to_excel(file_path, index=False, header=False)
            else:
                df.to_csv(file_path, index=False, header=False, sep=' ')

            logger.info(f"Arquivo de exemplo criado: {file_path}")
        except Exception as e:
            logger.error(f"Falha ao criar arquivo de exemplo: {e}")


def install_excel_deps():
    """
    Verifica e instala dependências para leitura de arquivos Excel se necessário.

    Retorna:
        bool: True se as dependências estão disponíveis, False caso contrário

    Exemplo:
      #  >>> if not install_excel_deps():
        ...     print("Não foi possível instalar dependências do Excel")
    """
    try:
        import openpyxl
        import xlrd
        return True
    except ImportError:
        try:
            print("Instalando dependências para leitura de Excel...")
            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                "openpyxl",
                "xlrd"
            ])
            return True
        except Exception as e:
            logger.error(f"Falha ao instalar dependências: {e}")
            return False


class LotteryStatistics:
    """
    Classe para cálculo de estatísticas básicas sobre resultados históricos.

    Atributos:
        results (List[List[int]]): Lista de jogos históricos
        all_numbers (np.ndarray): Array com todos os números possíveis (1-25)
        frequency (Counter): Frequência de cada número nos históricos
        delay (Dict[int, int]): Atraso atual de cada número (último sorteio)
        trends (Dict[int, str]): Tendência de cada número ("heating" ou "cooling")

    Métodos Principais:
        _calculate_frequency_and_delay(): Calcula frequência e atraso
        _identify_trends(): Identifica números em aquecimento/resfriamento
        get_geometric_features(): Extrai features geométricas de um jogo
    """

    def __init__(self, results: List[List[int]]):
        """
        Inicializa o objeto com resultados históricos e calcula estatísticas.

        Parâmetros:
            results (List[List[int]]): Lista de jogos históricos
        """
        self.results = results
        self.all_numbers = np.arange(1, LotteryConfig.TOTAL_NUMBERS + 1)
        self.frequency, self.delay = self._calculate_frequency_and_delay()
        self.trends = self._identify_trends()

    def _calculate_frequency_and_delay(self) -> Tuple[Counter, Dict[int, int]]:
        """
        Calcula frequência e atraso de cada número nos resultados históricos.

        Retorna:
            Tuple[Counter, Dict[int, int]]:
                - Counter com frequência de cada número
                - Dicionário com atraso atual de cada número

        Exemplo:
          #  >>> freq, delay = stats._calculate_frequency_and_delay()
          #  >>> print(f"Frequência do número 1: {freq[1]}")
          #  >>> print(f"Atraso do número 1: {delay[1]}")
        """
        frequency = Counter(np.concatenate(self.results))
        delay = {num: 0 for num in self.all_numbers}
        last_draw = np.zeros(LotteryConfig.TOTAL_NUMBERS + 1, dtype=bool)

        for game in reversed(self.results):
            last_draw.fill(False)
            last_draw[game] = True
            for num in self.all_numbers:
                delay[num] = 0 if last_draw[num] else delay[num] + 1

        return frequency, delay

    def _calculate_moving_average(self) -> Dict[int, float]:
        """
        Calcula média móvel de frequência para identificar tendências.

        Retorna:
            Dict[int, float]: Média móvel de frequência para cada número

        Nota:
            Usa TREND_WINDOW_SIZE da configuração para tamanho da janela
        """
        window = self.results[-LotteryConfig.TREND_WINDOW_SIZE:] \
            if len(self.results) > LotteryConfig.TREND_WINDOW_SIZE else self.results
        return {
            num: np.mean([num in game for game in window])
            for num in self.all_numbers
        }

    def _identify_trends(self) -> Dict[int, str]:
        """
        Identifica tendências de números (aquecendo/resfriando) comparando
        a frequência recente com a frequência histórica.

        Retorna:
            Dict[int, str]: "heating" se número está aquecendo, "cooling" caso contrário
        """
        moving_avg = self._calculate_moving_average()
        total_draws = len(self.results)
        return {
            num: "heating" if moving_avg[num] > self.frequency[num] / total_draws
            else "cooling"
            for num in self.all_numbers
        }

    def get_geometric_features(self, numbers: List[int]) -> List[float]:
        """
        Calcula features geométricas baseadas na posição dos números no volante.

        Parâmetros:
            numbers (List[int]): Lista de números a serem analisados

        Retorna:
            List[float]: Lista com features geométricas incluindo:
                - Coordenadas do centróide (x, y)
                - Dispersão dos pontos
                - Contagem por quadrante (4 valores)

        Exemplo:
           # >>> features = stats.get_geometric_features([1, 2, 3, 4, 5])
           # >>> print(f"Centróide: ({features[0]}, {features[1]})")
        """
        positions = [LotteryWheel.POSITIONS[num] for num in numbers]
        x, y = zip(*positions)

        centroid_x = np.mean(x)
        centroid_y = np.mean(y)
        dispersion = np.sqrt(np.var(x) + np.var(y))

        quadrants = np.zeros(4)
        for px, py in positions:
            if px <= 2 and py <= 2:
                quadrants[0] += 1
            elif px <= 2 and py > 2:
                quadrants[1] += 1
            elif px > 2 and py <= 2:
                quadrants[2] += 1
            else:
                quadrants[3] += 1

        return [centroid_x, centroid_y, dispersion, *quadrants]


class AdvancedStatistics(LotteryStatistics):
    """
    Estatísticas avançadas que herda de LotteryStatistics e adiciona:
    - Análise de pares consecutivos
    - Cálculo de correlações entre números
    - Análise sazonal
    - Clusterização de números

    Atributos Adicionais:
        consecutive_pairs (Dict[Tuple[int, int], int]): Contagem de pares consecutivos
        number_correlations (np.ndarray): Matriz de correlação entre números
        seasonal_trends (Dict[int, Dict[int, float]]): Tendências por janela temporal
        number_clusters (Dict[int, int]): Cluster de cada número baseado em co-ocorrência
    """

    def __init__(self, results: List[List[int]]):
        """
        Inicializa estatísticas avançadas chamando o construtor pai e
        calculando métricas adicionais.
        """
        super().__init__(results)
        self.consecutive_pairs = self._analyze_consecutive_pairs()
        self.number_correlations = self._calculate_correlations()
        self.seasonal_trends = self._analyze_seasonal_trends()
        self.number_clusters = self._cluster_numbers()

    def _analyze_consecutive_pairs(self) -> Dict[Tuple[int, int], int]:
        """
        Conta frequência de pares consecutivos nos jogos históricos.

        Retorna:
            Dict[Tuple[int, int], int]: Dicionário com contagem de cada par consecutivo

        Exemplo:
          #  >>> pairs = stats._analyze_consecutive_pairs()
          #  >>> print(f"Frequência do par (1,2): {pairs.get((1,2), 0)}")
        """
        pair_counts = Counter()
        for game in self.results:
            sorted_game = sorted(game)
            for i in range(len(sorted_game) - 1):
                pair = (sorted_game[i], sorted_game[i + 1])
                pair_counts[pair] += 1
        return pair_counts

    def _calculate_correlations(self) -> np.ndarray:
        """
        Calcula matriz de correlação entre números baseada em co-ocorrência.

        Retorna:
            np.ndarray: Matriz 25x25 de correlações entre números (0-1)

        Nota:
            Usa correlação de Pearson entre vetores de presença
        """
        presence_matrix = np.zeros((len(self.results), LotteryConfig.TOTAL_NUMBERS))
        for i, game in enumerate(self.results):
            presence_matrix[i, [num - 1 for num in game]] = 1
        return np.corrcoef(presence_matrix.T)

    def _analyze_seasonal_trends(self) -> Dict[int, Dict[int, float]]:
        """
        Analisa tendências sazonais em diferentes janelas temporais.

        Retorna:
            Dict[int, Dict[int, float]]: Dicionário aninhado com:
                - Chave externa: número (1-25)
                - Chave interna: tamanho da janela (10, 30, 100)
                - Valor: frequência normalizada na janela
        """
        seasonal_data = {num: {} for num in self.all_numbers}

        for window in EnhancedLotteryConfig.SEASONAL_WINDOWS:
            if len(self.results) < window:
                continue

            window_games = self.results[-window:]
            window_counts = Counter(np.concatenate(window_games))

            for num in self.all_numbers:
                seasonal_data[num][window] = window_counts.get(num, 0) / window

        return seasonal_data

    def _cluster_numbers(self) -> Dict[int, int]:
        """
        Agrupa números por padrões de co-ocorrência usando distância de correlação.

        Retorna:
            Dict[int, int]: Mapeamento número -> ID do cluster

        Nota:
            Usa limiar de 0.7 para definir clusters similares
        """
        presence_matrix = np.zeros((len(self.results), LotteryConfig.TOTAL_NUMBERS))
        for i, game in enumerate(self.results):
            presence_matrix[i, [num - 1 for num in game]] = 1

        # Calcula distâncias entre números (1 - correlação)
        distances = squareform(pdist(presence_matrix.T, 'correlation'))

        # Clusterização simples (k-means poderia ser usado aqui)
        clusters = {}
        cluster_id = 0
        threshold = 0.7  # Limiar de similaridade

        for num in self.all_numbers:
            if num in clusters:
                continue

            clusters[num] = cluster_id
            for other_num in self.all_numbers:
                if num != other_num and distances[num - 1][other_num - 1] < threshold:
                    clusters[other_num] = cluster_id

            cluster_id += 1

        return clusters

    def get_consecutive_score(self, numbers: List[int]) -> float:
        """
        Calcula score baseado em pares consecutivos frequentes no jogo.

        Parâmetros:
            numbers (List[int]): Lista de números do jogo

        Retorna:
            float: Score normalizado pelo tamanho do jogo

        Exemplo:
         #   >>> score = stats.get_consecutive_score([1, 2, 3, 4, 5])
         #  >>> print(f"Score de pares consecutivos: {score:.2f}")
        """
        sorted_numbers = sorted(numbers)
        score = 0

        for i in range(len(sorted_numbers) - 1):
            pair = (sorted_numbers[i], sorted_numbers[i + 1])
            score += self.consecutive_pairs.get(pair, 0)

        return score / len(numbers)

    def get_cluster_diversity(self, numbers: List[int]) -> float:
        """
        Calcula diversidade de clusters em um jogo.

        Parâmetros:
            numbers (List[int]): Lista de números do jogo

        Retorna:
            float: Proporção de clusters únicos no jogo (0-1)
        """
        clusters_in_game = {self.number_clusters[num] for num in numbers}
        return len(clusters_in_game) / len(set(self.number_clusters.values()))


class GameValidator:
    """
    Validador de jogos com regras básicas de distribuição.

    Métodos Estáticos:
        validate_distribution(): Verifica par/ímpar, baixo/alto e soma
        avoid_common_patterns(): Verifica padrões comuns no volante
    """

    @staticmethod
    def validate_distribution(numbers: List[int]) -> bool:
        """
        Verifica distribuição de pares/ímpares, baixos/altos e soma total.

        Parâmetros:
            numbers (List[int]): Lista de números a validar

        Retorna:
            bool: True se atende todos os critérios, False caso contrário

        Exemplo:
           # >>> valido = GameValidator.validate_distribution([1, 2, 3, ..., 15])
           # >>> print(f"Jogo válido: {valido}")
        """
        numbers_arr = np.array(numbers)
        even = np.sum(numbers_arr % 2 == 0)
        low = np.sum(numbers_arr <= 12)
        total_sum = np.sum(numbers_arr)

        return all([
            LotteryConfig.MIN_EVEN <= even <= LotteryConfig.MAX_EVEN,
            LotteryConfig.MIN_ODD <= (len(numbers) - even) <= LotteryConfig.MAX_ODD,
            LotteryConfig.MIN_LOW <= low <= LotteryConfig.MAX_LOW,
            LotteryConfig.MIN_SUM <= total_sum <= LotteryConfig.MAX_SUM
        ])

    @staticmethod
    def avoid_common_patterns(numbers: List[int]) -> bool:
        """
        Verifica se o jogo evita padrões comuns no volante.

        Parâmetros:
            numbers (List[int]): Lista de números a verificar

        Retorna:
            bool: True se não coincide com padrões comuns, False caso contrário
        """
        return set(numbers) not in LotteryWheel.COMMON_PATTERNS


class EnhancedGameValidator(GameValidator):
    """
    Validador avançado que herda de GameValidator e adiciona regras:
    - Distribuição por quadrantes
    - Diversidade de clusters
    - Correlação entre números

    Métodos Estáticos Adicionais:
        validate_quadrants(): Verifica distribuição mínima por quadrantes
        validate_cluster_diversity(): Garante diversidade de clusters
        validate_correlation(): Evita números altamente correlacionados
    """

    @staticmethod
    def validate_quadrants(numbers: List[int]) -> bool:
        """
        Verifica distribuição mínima de números por quadrantes do volante.

        Parâmetros:
            numbers (List[int]): Lista de números a validar

        Retorna:
            bool: True se cada quadrante tem pelo menos QUADRANT_MIN números
        """
        quadrants = [0, 0, 0, 0]
        for num in numbers:
            quad = AdvancedLotteryWheel.get_quadrant(num)
            if 1 <= quad <= 4:
                quadrants[quad - 1] += 1

        return all(q >= EnhancedLotteryConfig.QUADRANT_MIN for q in quadrants)

    @staticmethod
    def validate_cluster_diversity(numbers: List[int], stats: AdvancedStatistics) -> bool:
        """
        Verifica diversidade de clusters no jogo.

        Parâmetros:
            numbers (List[int]): Lista de números do jogo
            stats (AdvancedStatistics): Estatísticas com informação de clusters

        Retorna:
            bool: True se jogo tem números de pelo menos 3 clusters diferentes
        """
        clusters_in_game = {stats.number_clusters[num] for num in numbers}
        return len(clusters_in_game) >= 3

    @staticmethod
    def validate_correlation(numbers: List[int], stats: AdvancedStatistics) -> bool:
        """
        Verifica se o jogo não tem números altamente correlacionados.

        Parâmetros:
            numbers (List[int]): Lista de números do jogo
            stats (AdvancedStatistics): Estatísticas com matriz de correlação

        Retorna:
            bool: False se algum par tem correlação > 0.7, True caso contrário
        """
        for i, num1 in enumerate(numbers):
            for num2 in numbers[i + 1:]:
                if stats.number_correlations[num1 - 1][num2 - 1] > 0.7:
                    return False
        return True


class LotteryModel:
    """
    Modelo de machine learning básico (MLP) para previsão de jogos.

    Atributos:
        stats (LotteryStatistics): Estatísticas usadas para treino
        model (MLPClassifier): Modelo de rede neural treinado
        scaler (StandardScaler): Normalizador de features

    Métodos Principais:
        _prepare_training_data(): Prepara dados positivos e negativos
        _generate_features(): Extrai features de um conjunto de números
        _train_model(): Treina o modelo MLP
        predict_proba(): Prediz probabilidade de um jogo ser vencedor
    """

    def __init__(self, stats: LotteryStatistics):
        """
        Inicializa o modelo e realiza o treinamento.

        Parâmetros:
            stats (LotteryStatistics): Estatísticas para treino
        """
        self.stats = stats
        self.model, self.scaler = self._train_model()

    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara dados de treino balanceados com jogos reais (positivos)
        e jogos aleatórios válidos (negativos).

        Retorna:
            Tuple[np.ndarray, np.ndarray]: Features (X) e labels (y) balanceados
        """
        # Dados positivos (jogos reais)
        X_pos = np.array([self._generate_features(game) for game in self.stats.results])
        y_pos = np.ones(len(self.stats.results))

        # Dados negativos (jogos aleatórios válidos)
        X_neg = []
        for _ in range(LotteryConfig.NEGATIVE_SAMPLES_MULTIPLIER * len(self.stats.results)):
            game = random.sample(list(self.stats.all_numbers), LotteryConfig.NUMBERS_PER_GAME)
            while not GameValidator.validate_distribution(game):
                game = random.sample(list(self.stats.all_numbers), LotteryConfig.NUMBERS_PER_GAME)
            X_neg.append(self._generate_features(game))

        # Balanceia os dados
        X = np.vstack((X_pos, np.array(X_neg)))
        y = np.concatenate((y_pos, np.zeros(len(X_neg))))

        return resample(
            X, y,
            n_samples=min(len(X_pos), len(X_neg)) * 2,
            random_state=42
        )

    def _generate_features(self, numbers: List[int]) -> List[float]:
        """
        Gera features para um conjunto de números.

        Parâmetros:
            numbers (List[int]): Lista de números a serem transformados

        Retorna:
            List[float]: Lista de features incluindo:
                - Soma total
                - Contagem par/ímpar
                - Contagem baixo/alto
                - Atraso médio
                - Frequência média
                - Features geométricas
        """
        numbers_arr = np.array(numbers)
        even = np.sum(numbers_arr % 2 == 0)
        low = np.sum(numbers_arr <= 12)
        total_sum = np.sum(numbers_arr)
        avg_delay = np.mean([self.stats.delay[num] for num in numbers])
        avg_freq = np.mean([self.stats.frequency[num] for num in numbers])
        geometric = self.stats.get_geometric_features(numbers)

        return [total_sum, even, len(numbers) - even, low, len(numbers) - low,
                avg_delay, avg_freq, *geometric]

    def _train_model(self) -> Tuple[MLPClassifier, StandardScaler]:
        """
        Treina o modelo de rede neural com early stopping.

        Retorna:
            Tuple[MLPClassifier, StandardScaler]: Modelo treinado e normalizador
        """
        X, y = self._prepare_training_data()
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)

        model = MLPClassifier(
            hidden_layer_sizes=(50, 25),
            activation='relu',
            solver='adam',
            early_stopping=True,
            max_iter=1000,
            random_state=42
        )
        model.fit(X_scaled, y)

        return model, scaler

    def predict_proba(self, numbers: List[int]) -> float:
        """
        Prediz probabilidade de um conjunto de números ser vencedor.

        Parâmetros:
            numbers (List[int]): Lista de números a serem avaliados

        Retorna:
            float: Probabilidade estimada (0-1)

        Exemplo:
          #  >>> prob = model.predict_proba([1, 2, 3, ..., 15])
          #  >>> print(f"Probabilidade: {prob:.2%}")
        """
        features = self._generate_features(numbers)
        features_scaled = self.scaler.transform([features])
        return self.model.predict_proba(features_scaled)[0][1]


class LotteryModelEnsemble:
    """
    Ensemble avançado combinando MLP, Random Forest e XGBoost.

    Atributos:
        stats (AdvancedStatistics): Estatísticas para treino
        scaler (StandardScaler): Normalizador de features
        models (List): Lista dos 3 modelos do ensemble

    Métodos Principais:
        _generate_features(): Gera features completas com métricas avançadas
        _train_ensemble(): Treina os 3 modelos com dados balanceados
        predict_proba(): Combina predições dos modelos com pesos configuráveis
    """

    def __init__(self, stats: AdvancedStatistics):
        """
        Inicializa o ensemble e realiza o treinamento.

        Parâmetros:
            stats (AdvancedStatistics): Estatísticas avançadas para treino
        """
        self.stats = stats
        self.scaler = None
        self.models = []
        self._train_ensemble()

    def _generate_features(self, numbers: List[int]) -> List[float]:
        """
        Gera features completas para um conjunto de números incluindo:
        - Estatísticas básicas
        - Features geométricas
        - Métricas avançadas (pares consecutivos, clusters, sazonalidade)

        Parâmetros:
            numbers (List[int]): Lista de números a serem transformados

        Retorna:
            List[float]: Lista com 13 features calculadas

        Levanta:
            Exception: Se ocorrer erro no cálculo das features
        """
        try:
            # Features básicas
            numbers_arr = np.array(numbers)
            even = np.sum(numbers_arr % 2 == 0)
            low = np.sum(numbers_arr <= 12)
            total_sum = np.sum(numbers_arr)
            avg_delay = np.mean([self.stats.delay[num] for num in numbers])
            avg_freq = np.mean([self.stats.frequency[num] for num in numbers])

            # Features geométricas
            positions = [LotteryWheel.POSITIONS[num] for num in numbers]
            x, y = zip(*positions)
            centroid_x = np.mean(x)
            centroid_y = np.mean(y)
            dispersion = np.sqrt(np.var(x) + np.var(y))

            # Features avançadas
            consecutive_score = self._calculate_consecutive_score(numbers)
            cluster_diversity = self._calculate_cluster_diversity(numbers)
            seasonal_score = self._calculate_seasonal_score(numbers)

            return [
                total_sum,
                even,
                len(numbers) - even,
                low,
                len(numbers) - low,
                avg_delay,
                avg_freq,
                centroid_x,
                centroid_y,
                dispersion,
                consecutive_score,
                cluster_diversity,
                seasonal_score
            ]
        except Exception as e:
            logger.error(f"Erro ao gerar features: {e}")
            raise

    def _calculate_consecutive_score(self, numbers: List[int]) -> float:
        """Calcula score baseado em pares consecutivos frequentes"""
        sorted_numbers = sorted(numbers)
        score = 0.0
        for i in range(len(sorted_numbers) - 1):
            pair = (sorted_numbers[i], sorted_numbers[i + 1])
            score += self.stats.consecutive_pairs.get(pair, 0)
        return score / len(numbers)

    def _calculate_cluster_diversity(self, numbers: List[int]) -> float:
        """Calcula diversidade de clusters em um jogo (0-1)"""
        clusters = [self.stats.number_clusters[num] for num in numbers]
        unique_clusters = len(set(clusters))
        return unique_clusters / len(set(self.stats.number_clusters.values()))

    def _calculate_seasonal_score(self, numbers: List[int]) -> float:
        """Calcula score sazonal médio nas janelas configuradas"""
        score = 0.0
        count = 0
        for num in numbers:
            for window in EnhancedLotteryConfig.SEASONAL_WINDOWS:
                if window in self.stats.seasonal_trends[num]:
                    score += self.stats.seasonal_trends[num][window]
                    count += 1
        return score / count if count > 0 else 0.0

    def _train_ensemble(self):
        """Treina o ensemble com MLP, Random Forest e XGBoost"""
        try:
            # 1. Preparar dados
            X, y = self._prepare_training_data()

            # 2. Normalização
            self.scaler = StandardScaler().fit(X)
            X_scaled = self.scaler.transform(X)

            # 3. Configuração dos modelos
            self.models = [
                # MLP com 2 camadas ocultas
                MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    early_stopping=True,
                    max_iter=2000,
                    random_state=42
                ),
                # Random Forest com 200 árvores
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42
                ),
                # XGBoost com configuração otimizada
                xgb.XGBClassifier(
                    n_estimators=150,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
            ]

            # 4. Treino paralelo
            for model in self.models:
                model.fit(X_scaled, y)

            logger.info("Ensemble treinado com sucesso")

        except Exception as e:
            logger.error(f"Falha no treino do ensemble: {str(e)}")
            raise

    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara dados balanceados para treino do ensemble"""
        # Dados positivos (jogos reais)
        X_pos = np.array([self._generate_features(game) for game in self.stats.results])
        y_pos = np.ones(len(self.stats.results))

        # Dados negativos (jogos aleatórios válidos)
        X_neg = []
        for _ in range(LotteryConfig.NEGATIVE_SAMPLES_MULTIPLIER * len(self.stats.results)):
            game = self._generate_random_valid_game()
            X_neg.append(self._generate_features(game))

        # Balanceamento
        X = np.vstack((X_pos, np.array(X_neg)))
        y = np.concatenate((y_pos, np.zeros(len(X_neg))))

        return resample(X, y, n_samples=min(len(X_pos), len(X_neg)) * 2, random_state=42)

    def _generate_random_valid_game(self) -> List[int]:
        """Gera jogos aleatórios que atendem às regras básicas"""
        for _ in range(1000):  # Limite de tentativas
            game = random.sample(list(self.stats.all_numbers), LotteryConfig.NUMBERS_PER_GAME)
            if GameValidator.validate_distribution(game):
                return game
        raise ValueError("Não foi possível gerar jogo válido após 1000 tentativas")

    def predict_proba(self, numbers: List[int]) -> float:
        """
        Combina predições dos 3 modelos com pesos configuráveis.

        Parâmetros:
            numbers (List[int]): Lista de números a serem avaliados

        Retorna:
            float: Probabilidade média ponderada (0-1)

        Levanta:
            Exception: Se ocorrer erro durante a predição
        """
        try:
            features = self._generate_features(numbers)
            features_scaled = self.scaler.transform([features])

            probas = [model.predict_proba(features_scaled)[0][1] for model in self.models]
            return np.average(probas, weights=EnhancedLotteryConfig.MODEL_WEIGHTS)

        except Exception as e:
            logger.error(f"Erro na predição: {str(e)}")
            raise


class LotteryGenerator:
    """
    Gerador básico de jogos usando estatísticas e modelo.

    Atributos:
        stats (LotteryStatistics): Estatísticas para cálculo de scores
        model (LotteryModel): Modelo para estimar probabilidades

    Métodos Principais:
        generate_optimized_games(): Gera jogos otimizados
        _calculate_scores(): Calcula scores para cada número
        _select_numbers(): Seleciona números baseado nos scores
        _is_valid_game(): Verifica se um jogo é válido e único
    """

    def __init__(self, stats: LotteryStatistics, model: LotteryModel):
        """
        Inicializa o gerador com estatísticas e modelo.

        Parâmetros:
            stats (LotteryStatistics): Estatísticas históricas
            model (LotteryModel): Modelo treinado
        """
        self.stats = stats
        self.model = model

    def generate_optimized_games(self, num_games: int = LotteryConfig.GAMES_TO_GENERATE) -> List[List[int]]:
        """
        Gera jogos otimizados usando estatísticas e modelo.

        Parâmetros:
            num_games (int): Quantidade de jogos a gerar

        Retorna:
            List[List[int]]: Lista de jogos gerados

        Algoritmo:
            1. Pré-calcula features e probabilidades
            2. Para cada jogo a ser gerado:
                a. Calcula scores para cada número
                b. Seleciona números ponderados pelo score
                c. Valida o jogo gerado
        """
        max_freq = max(self.stats.frequency.values()) or 1
        games = []

        # Pré-calcula features e probabilidades
        features = np.array([
            self.model._generate_features([num]) for num in self.stats.all_numbers
        ])
        features_scaled = self.model.scaler.transform(features)
        probabilities = self.model.model.predict_proba(features_scaled)[:, 1]

        while len(games) < num_games:
            # Calcula scores para cada número
            scores = self._calculate_scores(max_freq, probabilities)

            # Seleciona números ponderados pelo score
            selected = self._select_numbers(scores)

            if self._is_valid_game(selected, games):
                games.append(selected)

        return games

    def _calculate_scores(self, max_freq: int, probabilities: np.ndarray) -> Dict[int, float]:
        """
        Calcula scores combinados para cada número.

        Parâmetros:
            max_freq (int): Frequência máxima para normalização
            probabilities (np.ndarray): Probabilidades do modelo

        Retorna:
            Dict[int, float]: Dicionário número -> score calculado

        Fórmula do Score:
            score = (freq_norm * FREQ_WEIGHT) +
                   (1/(delay+1) * delay_weight) +
                   (prob * PROB_WEIGHT) +
                   trend_weight
        """
        scores = {}
        for num in self.stats.all_numbers:
            # Peso do atraso (ajustado por faixa)
            delay = self.stats.delay[num]
            if delay <= LotteryConfig.LOW_DELAY_THRESHOLD:
                delay_weight = LotteryConfig.LOW_DELAY_WEIGHT
            elif delay <= LotteryConfig.MEDIUM_DELAY_THRESHOLD:
                delay_weight = LotteryConfig.MEDIUM_DELAY_WEIGHT
            else:
                delay_weight = LotteryConfig.HIGH_DELAY_WEIGHT

            # Peso da tendência
            trend_weight = LotteryConfig.TREND_WEIGHT if self.stats.trends[
                                                             num] == "heating" else -LotteryConfig.TREND_WEIGHT

            # Score final
            scores[num] = (
                    (self.stats.frequency[num] / max_freq) * LotteryConfig.FREQ_WEIGHT +
                    (1 / (delay + 1)) * delay_weight +
                    probabilities[num - 1] * LotteryConfig.PROB_WEIGHT +
                    trend_weight
            )
        return scores

    def _select_numbers(self, scores: Dict[int, float]) -> List[int]:
        """
        Seleciona números para um jogo usando amostragem ponderada.

        Parâmetros:
            scores (Dict[int, float]): Scores para cada número

        Retorna:
            List[int]: Lista de números selecionados

        Nota:
            - Considera apenas números com atraso <= MAX_DELAY
            - Converte scores em probabilidades com softmax
            - Amostra sem reposição
        """
        valid_numbers = [
            num for num in self.stats.all_numbers
            if self.stats.delay[num] <= LotteryConfig.MAX_DELAY
        ]
        valid_scores = [scores[num] for num in valid_numbers]

        # Converte scores em probabilidades com softmax
        exp_scores = np.exp(valid_scores - np.max(valid_scores))
        prob = exp_scores / np.sum(exp_scores)

        # Amostragem sem repetição
        selected = np.random.choice(
            valid_numbers,
            size=LotteryConfig.NUMBERS_PER_GAME,
            replace=False,
            p=prob
        )

        return sorted(selected)

    def _is_valid_game(self, game: List[int], existing_games: List[List[int]]) -> bool:
        """
        Verifica se um jogo é válido e único.

        Parâmetros:
            game (List[int]): Jogo a ser validado
            existing_games (List[List[int]]): Já gerados

        Retorna:
            bool: True se válido e único, False caso contrário
        """
        return (
                GameValidator.validate_distribution(game) and
                GameValidator.avoid_common_patterns(game) and
                game not in existing_games
        )


class EnhancedLotteryGenerator(LotteryGenerator):
    """
    Gerador avançado com múltiplas otimizações e validações.
    Herda de LotteryGenerator e adiciona:
    - Balanceamento em 4 níveis
    - Validações avançadas
    - Fallback para geração básica

    Atributos Adicionais:
        ensemble_model (LotteryModelEnsemble): Ensemble para predição
        base_model (LotteryModel): Modelo básico como fallback
        generation_attempts (int): Contador de tentativas

    Métodos Adicionais:
        _generate_initial_game(): Gera jogo inicial por scores
        _optimize_game_balance(): Aplica 4 níveis de otimização
        _is_valid_enhanced_game(): Validação com regras avançadas
    """

    def __init__(self, stats: AdvancedStatistics, model: LotteryModelEnsemble):
        """
        Inicializa o gerador avançado com estatísticas e ensemble.

        Parâmetros:
            stats (AdvancedStatistics): Estatísticas avançadas
            model (LotteryModelEnsemble): Ensemble treinado
        """
        self.stats = stats
        self.ensemble_model = model
        self.base_model = LotteryModel(stats)
        self.generation_attempts = 0

    def generate_optimized_games(self, num_games: int = LotteryConfig.GAMES_TO_GENERATE) -> List[List[int]]:
        """
        Gera jogos otimizados com todas as validações e otimizações.

        Parâmetros:
            num_games (int): Quantidade de jogos a gerar

        Retorna:
            List[List[int]]: Lista de jogos gerados

        Algoritmo:
            1. Geração inicial baseada em scores híbridos
            2. Otimização em 4 níveis:
               a. Balanceamento par/ímpar
               b. Balanceamento baixo/alto
               c. Distribuição por quadrantes
               d. Verificação final
            3. Se falhar, usa fallback básico
        """
        games = []
        attempts = 0
        max_attempts = num_games * 50

        while len(games) < num_games and attempts < max_attempts:
            attempts += 1
            try:
                # Geração inicial
                game = self._generate_initial_game()

                # Otimização em múltiplos níveis
                optimized_game = self._optimize_game_balance(game)

                # Validação final
                if optimized_game and self._is_valid_enhanced_game(optimized_game, games):
                    games.append(optimized_game)
                    logger.info(f"Jogo {len(games)}/{num_games} gerado")
                    attempts = 0  # Reset do contador após sucesso

            except Exception as e:
                logger.debug(f"Tentativa {attempts} falhou: {str(e)}")
                continue

        # Fallback garantido
        if len(games) < num_games:
            missing = num_games - len(games)
            logger.warning(f"Gerando {missing} jogos via método básico (fallback)")
            games.extend(self._generate_fallback_games(missing))

        return games

    def _generate_initial_game(self) -> List[int]:
        """Gera jogo inicial baseado em scores híbridos"""
        scores = self._calculate_hybrid_scores()
        valid_numbers = [
            num for num in self.stats.all_numbers
            if self.stats.delay[num] <= LotteryConfig.MAX_DELAY
        ]
        valid_scores = [scores[num] for num in valid_numbers]

        # Conversão para probabilidades com softmax
        exp_scores = np.exp(valid_scores - np.max(valid_scores))
        prob = exp_scores / np.sum(exp_scores)

        # Seleção sem repetição
        selected = np.random.choice(
            valid_numbers,
            size=LotteryConfig.NUMBERS_PER_GAME,
            replace=False,
            p=prob
        )

        return sorted(selected.tolist())

    def _calculate_hybrid_scores(self) -> Dict[int, float]:
        """Calcula scores combinando ensemble, atraso, frequência e tendência"""
        scores = {}
        for num in self.stats.all_numbers:
            # Probabilidade do modelo ensemble
            ensemble_prob = self.ensemble_model.predict_proba([num])

            # Fatores estatísticos
            delay = self.stats.delay[num]
            freq = self.stats.frequency[num] / len(self.stats.results)
            trend = 0.1 if self.stats.trends[num] == "heating" else -0.05

            # Score final ponderado
            scores[num] = (
                    0.4 * ensemble_prob +
                    0.3 * (1 / (delay + 1)) +
                    0.2 * freq +
                    0.1 * trend
            )
        return scores

    def _optimize_game_balance(self, game: List[int]) -> Optional[List[int]]:
        """
        Aplica 4 níveis de otimização ao jogo:
        1. Balanceamento par/ímpar
        2. Balanceamento baixo/alto
        3. Distribuição por quadrantes
        4. Verificação final

        Parâmetros:
            game (List[int]): Jogo a ser otimizado

        Retorna:
            Optional[List[int]]: Jogo otimizado ou None se falhar
        """
        optimized = game.copy()

        # 1. Balanceamento par/ímpar
        optimized = self._balance_even_odd(optimized)
        if not optimized:
            return None

        # 2. Balanceamento baixo/alto
        optimized = self._balance_low_high(optimized)
        if not optimized:
            return None

        # 3. Distribuição por quadrantes
        optimized = self._balance_quadrants(optimized)
        if not optimized:
            return None

        # 4. Verificação final
        return optimized if self._is_valid_basic_game(optimized) else None

    def _balance_even_odd(self, game: List[int]) -> Optional[List[int]]:
        """Garante balanceamento entre pares e ímpares"""
        even_count = sum(1 for num in game if num % 2 == 0)

        # Ajusta se estiver fora dos limites configurados
        if even_count < LotteryConfig.MIN_EVEN:
            return self._adjust_number_type(game, target_even=True)
        elif even_count > LotteryConfig.MAX_EVEN:
            return self._adjust_number_type(game, target_even=False)

        return game

    def _balance_low_high(self, game: List[int]) -> Optional[List[int]]:
        """Garante balanceamento entre números baixos e altos"""
        low_count = sum(1 for num in game if num <= 12)

        # Ajusta se estiver fora dos limites
        if low_count < LotteryConfig.MIN_LOW:
            return self._adjust_number_range(game, target_low=True)
        elif low_count > LotteryConfig.MAX_LOW:
            return self._adjust_number_range(game, target_low=False)

        return game

    def _balance_quadrants(self, game: List[int]) -> Optional[List[int]]:
        """Garante distribuição mínima por quadrantes"""
        quad_counts = Counter(AdvancedLotteryWheel.get_quadrant(num) for num in game)

        # Ajusta quadrantes com menos de 2 números
        for q in range(1, 5):
            if quad_counts.get(q, 0) < 2:
                adjusted = self._adjust_quadrant(game, q)
                if not adjusted or not self._is_valid_basic_game(adjusted):
                    return None
                return adjusted

        return game

    def _adjust_number_type(self, game: List[int], target_even: bool) -> Optional[List[int]]:
        """Substitui números para balancear pares/ímpares"""
        current_numbers = set(game)
        candidates = [
            n for n in self.stats.all_numbers
            if n not in current_numbers
               and (n % 2 == 0) == target_even
               and self.stats.delay[n] <= LotteryConfig.MAX_DELAY
        ]

        if not candidates:
            return None

        # Encontra o pior número do tipo oposto para substituir
        to_replace = [n for n in game if (n % 2 == 0) != target_even]
        if not to_replace:
            return game

        replace_num = min(to_replace, key=lambda x: self._calculate_number_score(x))
        new_num = max(candidates, key=lambda x: self._calculate_number_score(x))

        return sorted([n if n != replace_num else new_num for n in game])

    def _adjust_number_range(self, game: List[int], target_low: bool) -> Optional[List[int]]:
        """Substitui números para balancear baixos/altos"""
        current_numbers = set(game)
        threshold = 12
        candidates = [
            n for n in self.stats.all_numbers
            if n not in current_numbers
               and (n <= threshold) == target_low
               and self.stats.delay[n] <= LotteryConfig.MAX_DELAY
        ]

        if not candidates:
            return None

        # Encontra o pior número do range oposto
        to_replace = [n for n in game if (n <= threshold) != target_low]
        if not to_replace:
            return game

        replace_num = min(to_replace, key=lambda x: self._calculate_number_score(x))
        new_num = max(candidates, key=lambda x: self._calculate_number_score(x))

        return sorted([n if n != replace_num else new_num for n in game])

    def _adjust_quadrant(self, game: List[int], target_quad: int) -> Optional[List[int]]:
        """Substitui números para melhorar distribuição por quadrantes"""
        current_numbers = set(game)
        candidates = [
            n for n in self.stats.all_numbers
            if n not in current_numbers
               and AdvancedLotteryWheel.get_quadrant(n) == target_quad
               and self.stats.delay[n] <= LotteryConfig.MAX_DELAY
        ]

        if not candidates:
            return None

        # Encontra o quadrante com excesso
        quad_counts = Counter(AdvancedLotteryWheel.get_quadrant(n) for n in game)
        over_quad = max(quad_counts.keys(), key=lambda x: quad_counts[x])

        # Substitui o pior número do quadrante com excesso
        to_replace = [n for n in game if AdvancedLotteryWheel.get_quadrant(n) == over_quad]
        if not to_replace:
            return game

        replace_num = min(to_replace, key=lambda x: self._calculate_number_score(x))
        new_num = max(candidates, key=lambda x: self._calculate_number_score(x))

        return sorted([n if n != replace_num else new_num for n in game])

    def _calculate_number_score(self, number: int) -> float:
        """Calcula o score completo de um número individual"""
        ensemble_prob = self.ensemble_model.predict_proba([number])
        delay = self.stats.delay[number]
        freq = self.stats.frequency[number] / len(self.stats.results)
        trend = 0.1 if self.stats.trends[number] == "heating" else -0.05

        return (
                0.4 * ensemble_prob +
                0.3 * (1 / (delay + 1)) +
                0.2 * freq +
                0.1 * trend
        )

    def _is_valid_basic_game(self, game: List[int]) -> bool:
        """Validação básica do jogo (tamanho, distribuição, padrões)"""
        return (
                len(game) == LotteryConfig.NUMBERS_PER_GAME and
                GameValidator.validate_distribution(game) and
                GameValidator.avoid_common_patterns(game)
        )

    def _is_valid_enhanced_game(self, game: List[int], existing_games: List[List[int]]) -> bool:
        """Validação avançada com todas as regras (quadrantes, clusters, etc.)"""
        if not self._is_valid_basic_game(game) or game in existing_games:
            return False

        # Verificação de quadrantes
        quadrants = [AdvancedLotteryWheel.get_quadrant(num) for num in game]
        if any(quadrants.count(q) < 2 for q in range(1, 5)):
            return False

        # Verificação de clusters
        clusters = {self.stats.number_clusters[num] for num in game}
        if len(clusters) < 3:
            return False

        return True

    def _generate_fallback_games(self, num_games: int) -> List[List[int]]:
        """Geração básica de jogos como fallback quando a otimização falha"""
        games = []
        attempts = 0

        while len(games) < num_games and attempts < num_games * 10:
            attempts += 1
            game = random.sample(list(self.stats.all_numbers), LotteryConfig.NUMBERS_PER_GAME)
            if self._is_valid_basic_game(game) and game not in games:
                games.append(game)

        return games

    def display_advanced_stats(self, games: List[List[int]]) -> None:
        """
        Exibe estatísticas detalhadas dos jogos gerados.

        Parâmetros:
            games (List[List[int]]): Lista de jogos a serem analisados

        Saída:
            Imprime tabela com:
            - Número do jogo
            - Contagem de pares/ímpares
            - Contagem de baixos
            - Soma total
            - Distribuição por quadrantes
        """
        print("\nEstatísticas Avançadas dos Jogos:")
        print("{:<10} {:<8} {:<8} {:<8} {:<10} {:<12}".format(
            "Jogo", "Pares", "Ímpares", "Baixos", "Soma", "Quadrantes"))

        for i, game in enumerate(games, 1):
            even = sum(1 for num in game if num % 2 == 0)
            low = sum(1 for num in game if num <= 12)
            total = sum(game)
            quadrants = [AdvancedLotteryWheel.get_quadrant(num) for num in game]
            quad_dist = ",".join(f"{q}:{quadrants.count(q)}" for q in range(1, 5))

            print("{:<10} {:<8} {:<8} {:<8} {:<10} {:<12}".format(
                f"Jogo {i}", even, len(game) - even, low, total, quad_dist))


class LotteryApp:
    """
    Aplicação principal que orquestra todo o processo.

    Atributos:
        results (List[List[int]]): Resultados históricos carregados
        stats (AdvancedStatistics): Estatísticas calculadas
        model (LotteryModelEnsemble): Modelo treinado
        generator (EnhancedLotteryGenerator): Gerador de jogos
        file_path (str): Caminho do arquivo carregado

    Métodos Principais:
        load_data(): Carrega dados históricos
        initialize_components(): Prepara estatísticas e modelos
        generate_games(): Gera jogos otimizados
        display_games(): Exibe jogos formatados
    """

    def __init__(self):
        """Inicializa a aplicação com atributos vazios"""
        self.results = []
        self.stats = None
        self.model = None
        self.generator = None
        self.file_path = None

    def load_data(self, file_path: str) -> bool:
        """
        Carrega dados históricos de loteria com tratamento robusto.

        Parâmetros:
            file_path (str): Caminho do arquivo com os dados

        Retorna:
            bool: True se os dados foram carregados com sucesso

        Tratamento Especial:
            - Verifica existência do arquivo
            - Processa Excel e texto
            - Valida formato dos dados
            - Remove linhas inválidas
        """
        try:
            # Normaliza o caminho do arquivo
            file_path = os.path.normpath(file_path)

            if not os.path.exists(file_path):
                logger.error(f"Arquivo não encontrado: {file_path}")
                return False

            # Limpa resultados anteriores
            self.results = []

            # Processa arquivos Excel
            if file_path.lower().endswith(('.xlsx', '.xls')):
                try:
                    if not install_excel_deps():
                        logger.error("Dependências para Excel não disponíveis")
                        return False

                    df = pd.read_excel(file_path, header=None, engine='openpyxl')

                    for _, row in df.iterrows():
                        numbers = row.dropna().astype(int).tolist()
                        if len(numbers) == LotteryConfig.NUMBERS_PER_GAME:
                            self.results.append(numbers)
                        else:
                            logger.warning(f"Linha ignorada - quantidade inválida de números: {numbers}")

                except Exception as e:
                    logger.error(f"Erro ao processar arquivo Excel: {e}")
                    return False

            # Processa arquivos texto
            elif file_path.lower().endswith('.txt'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                numbers = list(map(int, line.strip().replace(',', ' ').split()))
                                if len(numbers) == LotteryConfig.NUMBERS_PER_GAME:
                                    self.results.append(numbers)
                                else:
                                    logger.warning(f"Linha ignorada - quantidade inválida de números: {numbers}")

                except Exception as e:
                    logger.error(f"Erro ao processar arquivo texto: {e}")
                    return False

            else:
                logger.error(f"Formato de arquivo não suportado: {file_path}")
                return False

            if not self.results:
                logger.error("Nenhum dado válido encontrado no arquivo")
                return False

            logger.info(f"Carregados {len(self.results)} jogos históricos")
            return True

        except Exception as e:
            logger.error(f"Erro inesperado ao carregar dados: {e}")
            return False

    def initialize_components(self) -> bool:
        """
        Inicializa todos os componentes necessários para geração.

        Retorna:
            bool: True se inicializado com sucesso

        Componentes Inicializados:
            1. AdvancedStatistics: Cálculo de estatísticas avançadas
            2. LotteryModelEnsemble: Treinamento do ensemble
            3. EnhancedLotteryGenerator: Preparação do gerador
        """
        if not self.results:
            logger.error("Dados não carregados. Execute load_data() primeiro")
            return False

        try:
            self.stats = AdvancedStatistics(self.results)
            self.model = LotteryModelEnsemble(self.stats)
            self.generator = EnhancedLotteryGenerator(self.stats, self.model)
            logger.info("Componentes inicializados com sucesso")
            return True
        except Exception as e:
            logger.error(f"Erro ao inicializar componentes: {e}")
            return False

    def generate_games(self, num_games: int = None) -> List[List[int]]:
        """
        Gera jogos otimizados com tratamento de erros.

        Parâmetros:
            num_games (int): Quantidade de jogos a gerar (opcional)

        Retorna:
            List[List[int]]: Lista de jogos gerados ou lista vazia se falhar
        """
        if not all([self.stats, self.model, self.generator]):
            logger.error("Componentes não inicializados. Execute initialize_components() primeiro")
            return []

        num_games = num_games or LotteryConfig.GAMES_TO_GENERATE
        try:
            games = self.generator.generate_optimized_games(num_games)
            logger.info(f"{len(games)} jogos gerados com sucesso")
            return games
        except Exception as e:
            logger.error(f"Erro ao gerar jogos: {e}")
            return []

    def display_games(self, games: List[List[int]]) -> None:
        """
        Exibe jogos formatados exatamente como solicitado.

        Parâmetros:
            games (List[List[int]]): Lista de jogos a serem exibidos

        Saída:
            Imprime cada jogo no formato:
            Sequencia 1: [ 1, 2, 3, ..., 15 ]
            Sequencia 2: [ 2, 4, 6, ..., 16 ]
            ...
        """
        if not games:
            print("Nenhum jogo para exibir")
            return

        print("\nJogos gerados:")
        for i, game in enumerate(games, 1):
            formatted_numbers = [f"{num:2}" for num in sorted(game)]
            print(f"Sequencia {i}: [ {', '.join(formatted_numbers)} ]")


def check_dependencies():
    """
    Verifica e instala todas as dependências necessárias.

    Dependências Verificadas:
        - numpy, pandas
        - scikit-learn, xgboost
        - openpyxl, xlrd (para Excel)
        - tkinter (interface)
        - scipy (cálculos científicos)
    """
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scikit-learn': 'scikit-learn',
        'xgboost': 'xgboost',
        'openpyxl': 'openpyxl',
        'xlrd': 'xlrd',
        'tkinter': 'python-tk',
        'scipy': 'scipy'
    }

    print("Verificando dependências...")
    for lib, pkg in required.items():
        try:
            __import__(lib)
            print(f"✓ {lib} instalado")
        except ImportError:
            print(f"Instalando {lib}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                print(f"✓ {lib} instalado com sucesso")
            except subprocess.CalledProcessError:
                print(f"✗ Falha ao instalar {lib}")


def main():
    """
    Função principal com tratamento completo de erros.

    Fluxo:
    1. Verifica dependências
    2. Cria instância da aplicação
    3. Seleciona e carrega arquivo
    4. Inicializa componentes
    5. Gera e exibe jogos
    6. Trata possíveis erros
    """
    print("=== SISTEMA AVANÇADO DE GERAÇÃO DE JOGOS DE LOTERIA ===")

    try:
        # Verifica dependências
        check_dependencies()

        app = LotteryApp()

        # 1. Seleção do arquivo
        print("\nSelecione o arquivo com resultados históricos...")
        file_path = FileHandler.select_file()

        if not file_path:
            print("\nNenhum arquivo selecionado. Criando arquivo de exemplo...")
            FileHandler.create_example_file("resultados_exemplo.xlsx")
            print("Arquivo de exemplo criado: 'resultados_exemplo.xlsx'")
            print("Por favor, preencha com seus dados e execute novamente.")
            return

        # 2. Carregamento de dados
        print(f"\nCarregando dados de: {file_path}")
        if not app.load_data(file_path):
            print("\nERRO: Não foi possível carregar os dados.")
            return

        # 3. Inicialização do sistema
        print("\nInicializando componentes avançados...")
        if not app.initialize_components():
            print("\nERRO: Falha ao inicializar o sistema.")
            return

        # 4. Geração de jogos
        print("\nGerando jogos otimizados com técnicas avançadas...")
        games = app.generate_games()

        if not games:
            print("\nERRO: Nenhum jogo foi gerado.")
        else:
            app.display_games(games)

    except Exception as e:
        logger.exception("Erro fatal durante a execução")
        print(f"\nERRO INESPERADO: {str(e)}")
    finally:
        input("\nPressione Enter para sair...")


if __name__ == "__main__":
    main()