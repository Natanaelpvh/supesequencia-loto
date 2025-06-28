"""
MÓDULO PRINCIPAL: SISTEMA DE GERAÇÃO DE JOGOS DE LOTERIA AVANÇADO

Este sistema implementa um gerador inteligente de jogos de loteria que combina:
- Análise estatística de resultados históricos
- Modelos de machine learning (ensemble)
- Validação baseada em regras probabilísticas
- Otimização geométrica no volante

Autor: Natanael
Versão: 1.0.1
Data: 14/06/2025
"""

# Bibliotecas padrão
import os
import sys
import random
import math
import pickle
import logging
import subprocess
from collections import Counter
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Tuple, Optional, Set
from multiprocessing import get_context

# Interface gráfica
import tkinter as tk
from tkinter import filedialog

# Matemática e dados
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# Machine Learning
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

# Visualização
import matplotlib.pyplot as plt

# Processamento paralelo
from joblib import Parallel, delayed

# XGBoost (instalação separada)
import xgboost as xgb

# Configuração para suprimir warnings específicos
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

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

    # Ajuste nos pesos para valorizar mais a diversificação
    FREQ_WEIGHT = 0.30  # Reduzido de 0.40
    DELAY_WEIGHT = 0.40  # Aumentado de 0.35
    PROB_WEIGHT = 0.20  # Reduzido de 0.25
    TREND_WEIGHT = 0.10  # Mantido
    GEOMETRY_WEIGHT = 0.15  # Novo peso para distribuição geométrica

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
    """
       Configurações estendidas com novos parâmetros avançados.
       Herda todos os atributos de LotteryConfig e adiciona:
       """
    REQUIRED_ADVANCED_CHECKS = 2
    QUADRANT_MIN = 2
    MAX_GENERATION_ATTEMPTS = 100

    # Novos parâmetros para análise avançada
    CONSECUTIVE_PAIR_WEIGHT = 0.15
    CORRELATION_PENALTY = 0.2
    SA_TEMPERATURE = 1.0
    SA_COOLING_RATE = 0.95
    SA_ITERATIONS = 100

    # Parâmetros do ensemble
    MODEL_WEIGHTS = [0.4, 0.3, 0.3]  # Pesos para MLP, RF, XGBoost

    # Configurações de análise temporal
    SEASONAL_WINDOWS = [10, 30, 100]  # Janelas para análise sazonal
    TREND_SENSITIVITY = 0.25

    # Novos parâmetros
    MODEL_CACHE_DIR = "model_cache"
    PLOT_OUTPUT_DIR = "game_plots"
    MAX_CACHED_GAMES = 1000
    NUM_PARALLEL_JOBS = 4


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
    - Análise de ciclos com FFT (NOVO)
    - Distribuição por quadrantes (NOVO)
    - Análise de números primos (NOVO)

    Atributos Adicionais:
        consecutive_pairs (Dict[Tuple[int, int], int]): Contagem de pares consecutivos
        number_correlations (np.ndarray): Matriz de correlação entre números
        seasonal_trends (Dict[int, Dict[int, float]]): Tendências por janela temporal
        number_clusters (Dict[int, int]): Cluster de cada número baseado em co-ocorrência
        prime_numbers (Set[int]): Conjunto de números primos na Lotofácil (NOVO)
        cycles (Dict[int, Dict[int, float]]): Padrões cíclicos por número (NOVO)
        quadrant_distributions (Dict[int, Dict[int, float]]): Distribuição por quadrante (NOVO)
    """

    def __init__(self, results: List[List[int]]):
        """Inicializa todas as análises, incluindo as novas"""
        super().__init__(results)
        # Métodos existentes
        self.consecutive_pairs = self._analyze_consecutive_pairs()
        self.number_correlations = self._calculate_correlations()
        self.seasonal_trends = self._analyze_seasonal_trends()
        self.number_clusters = self._cluster_numbers()

        # Novos atributos e análises
        self.prime_numbers = {2, 3, 5, 7, 11, 13, 17, 19, 23}
        self.cycles = self._detect_number_cycles()
        self.quadrant_distributions = self._calculate_quadrant_distributions()

    # --- Métodos existentes mantidos exatamente como estão ---
    def _analyze_consecutive_pairs(self) -> Dict[Tuple[int, int], int]:
        pair_counts = Counter()
        for game in self.results:
            sorted_game = sorted(game)
            for i in range(len(sorted_game) - 1):
                pair = (sorted_game[i], sorted_game[i + 1])
                pair_counts[pair] += 1
        return pair_counts

    def _calculate_correlations(self) -> np.ndarray:
        presence_matrix = np.zeros((len(self.results), LotteryConfig.TOTAL_NUMBERS))
        for i, game in enumerate(self.results):
            presence_matrix[i, [num - 1 for num in game]] = 1
        return np.corrcoef(presence_matrix.T)

    def _analyze_seasonal_trends(self) -> Dict[int, Dict[int, float]]:
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
        presence_matrix = np.zeros((len(self.results), LotteryConfig.TOTAL_NUMBERS))
        for i, game in enumerate(self.results):
            presence_matrix[i, [num - 1 for num in game]] = 1
        distances = squareform(pdist(presence_matrix.T, 'correlation'))
        clusters = {}
        cluster_id = 0
        threshold = 0.7
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
        sorted_numbers = sorted(numbers)
        score = 0
        for i in range(len(sorted_numbers) - 1):
            pair = (sorted_numbers[i], sorted_numbers[i + 1])
            score += self.consecutive_pairs.get(pair, 0)
        return score / len(numbers)

    def get_cluster_diversity(self, numbers: List[int]) -> float:
        clusters_in_game = {self.number_clusters[num] for num in numbers}
        return len(clusters_in_game) / len(set(self.number_clusters.values()))

    # --- Novos métodos adicionados ---
    def _detect_number_cycles(self) -> Dict[int, Dict[int, float]]:
        """Detecta padrões cíclicos no aparecimento de cada número usando FFT"""
        from scipy import fftpack

        cycles = {}
        for num in self.all_numbers:
            presence = [int(num in game) for game in self.results]
            n = len(presence)
            if n < 10:  # Mínimo de dados para análise
                cycles[num] = {}
                continue

            fft = fftpack.fft(presence)
            freqs = fftpack.fftfreq(n)

            num_cycles = {}
            for i in range(1, n // 2):
                if freqs[i] > 0:
                    period = int(1 / freqs[i])
                    if 2 <= period <= 50:
                        num_cycles[period] = abs(fft[i]) / n

            cycles[num] = dict(sorted(num_cycles.items(),
                                      key=lambda x: x[1], reverse=True)[:3])
        return cycles

    def _calculate_quadrant_distributions(self) -> Dict[int, Dict[int, float]]:
        """Calcula distribuição histórica de números por quadrantes"""
        quadrant_counts = {q: Counter() for q in range(1, 5)}
        for game in self.results:
            for num in game:
                quad = AdvancedLotteryWheel.get_quadrant(num)
                if 1 <= quad <= 4:
                    quadrant_counts[quad][num] += 1

        total_games = len(self.results)
        return {
            q: {num: count / total_games for num, count in counts.items()}
            for q, counts in quadrant_counts.items()
        }

    def get_enhanced_features(self, numbers: List[int]) -> List[float]:
        """
        Gera features estendidas incluindo as novas métricas
        Mantém compatibilidade com o sistema existente
        """
        base_features = super().get_geometric_features(numbers)

        # Features básicas
        numbers_arr = np.array(numbers)
        even = np.sum(numbers_arr % 2 == 0)
        low = np.sum(numbers_arr <= 12)
        total_sum = np.sum(numbers_arr)
        avg_delay = np.mean([self.delay[num] for num in numbers])
        avg_freq = np.mean([self.frequency[num] for num in numbers])

        # Novas features
        prime_ratio = sum(1 for num in numbers if num in self.prime_numbers) / len(numbers)
        sorted_nums = sorted(numbers)
        avg_gap = np.mean([sorted_nums[i + 1] - sorted_nums[i]
                           for i in range(len(sorted_nums) - 1)])

        # Entropia da distribuição
        hist = np.histogram(numbers, bins=5, range=(1, 25))[0]
        hist = hist / hist.sum()
        entropy = -np.sum([p * np.log(p) for p in hist if p > 0])

        # Features de ciclos
        current_draw = len(self.results) + 1
        cycle_strength = sum(
            max(self.cycles[num].values(), default=0)
            for num in numbers
        )

        # Features de quadrantes
        quadrants = [AdvancedLotteryWheel.get_quadrant(num) for num in numbers]
        quadrant_score = sum(
            self.quadrant_distributions[q].get(num, 0)
            for q, num in zip(quadrants, numbers) if 1 <= q <= 4
        )

        return [*base_features, total_sum, even, len(numbers) - even, low,
                len(numbers) - low, avg_delay, avg_freq, prime_ratio,
                avg_gap, entropy, cycle_strength, quadrant_score]

    def get_number_cycle_strength(self, number: int, current_draw: int) -> float:
        """Retorna a força do ciclo para um número no concurso atual"""
        return sum(
            strength for period, strength in self.cycles.get(number, {}).items()
            if current_draw % period == 0
        )

    def get_quadrant_distribution(self, quadrant: int) -> Dict[int, float]:
        """Retorna a distribuição histórica de um quadrante específico"""
        return self.quadrant_distributions.get(quadrant, {})

    @lru_cache(maxsize=EnhancedLotteryConfig.MAX_CACHED_GAMES)
    def get_enhanced_features_cached(self, numbers_tuple: Tuple[int]) -> List[float]:
        """Versão com cache do método get_enhanced_features"""
        return self.get_enhanced_features(list(numbers_tuple))

    def analyze_sequences(self, numbers: List[int]) -> Dict[str, float]:
        """
        Analisa sequências numéricas no jogo.
        Retorna métricas como:
        - Maior sequência consecutiva
        - Razão de Fibonacci
        - Progressões aritméticas
        """
        sorted_nums = sorted(numbers)
        gaps = [sorted_nums[i + 1] - sorted_nums[i] for i in range(len(sorted_nums) - 1)]

        # Calcula a maior sequência consecutiva
        max_consec = 1
        current = 1
        for i in range(1, len(sorted_nums)):
            if sorted_nums[i] == sorted_nums[i - 1] + 1:
                current += 1
                max_consec = max(max_consec, current)
            else:
                current = 1

        # Calcula razão de Fibonacci
        fib_pairs = [(sorted_nums[i], sorted_nums[i + 1])
                     for i in range(len(sorted_nums) - 1)
                     if sorted_nums[i + 1] / sorted_nums[i] > 1.6]
        fib_ratio = len(fib_pairs) / (len(sorted_nums) - 1) if len(sorted_nums) > 1 else 0

        return {
            'max_consecutive': max_consec,
            'fibonacci_ratio': fib_ratio,
            'avg_gap': np.mean(gaps),
            'gap_variation': np.std(gaps)
        }

class LotteryModelEnsemble:
    """
    Versão final com tratamento robusto de NaN e problemas de predição
    """

    def __init__(self, stats: AdvancedStatistics):
        self.stats = stats
        self.models = []
        self.model_weights = EnhancedLotteryConfig.MODEL_WEIGHTS
        self.scaler = None
        self.loaded_from_cache = False
        self.feature_names = [
            'total_sum', 'even_count', 'odd_count', 'low_count', 'high_count',
            'avg_delay', 'avg_freq', 'centroid_x', 'centroid_y', 'dispersion',
            'quad1', 'quad2', 'quad3', 'quad4', 'prime_ratio', 'avg_gap',
            'entropy', 'cycle_strength', 'quadrant_score'
        ]

        try:
            if not self._try_load_from_cache():
                self._train_with_fallbacks()
            self._validate_initial_models()
        except Exception as e:
            logger.critical(f"Falha crítica: {e}")
            self._emergency_fallback()

    def _try_load_from_cache(self) -> bool:
        """Tenta carregar modelos do cache. Retorna True se bem-sucedido."""
        try:
            cached_ensemble = LotteryModelEnsemble.load_models(self.stats)
            if cached_ensemble is not None:
                self.models = cached_ensemble.models
                self.scaler = cached_ensemble.scaler
                self.loaded_from_cache = True
                logger.info("Modelos carregados do cache com sucesso")
                return True
            return False
        except Exception as e:
            logger.warning(f"Falha ao carregar do cache: {e}")
            return False

    def _train_with_fallbacks(self):
        """Treinamento com tentativas de fallback se modelos falharem."""
        try:
            X, y = self._prepare_training_data()
            self.scaler = StandardScaler().fit(X)
            X_scaled = self.scaler.transform(X)

            # Inicializa modelos
            self.models = [
                self._create_rf(),
                self._create_mlp(),
                self._create_xgb()
            ]

            self._train_serial(X_scaled, y)
        except Exception as e:
            logger.error(f"Falha no treinamento principal: {e}")
            self._emergency_fallback(X_scaled, y)

    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara dados de treino com tratamento de NaN"""
        try:
            # Dados positivos
            X_pos = np.array([self._generate_features_safe(game) for game in self.stats.results])
            y_pos = np.ones(len(self.stats.results))

            # Dados negativos
            X_neg = []
            for _ in range(LotteryConfig.NEGATIVE_SAMPLES_MULTIPLIER * len(self.stats.results)):
                game = random.sample(list(self.stats.all_numbers), LotteryConfig.NUMBERS_PER_GAME)
                while not GameValidator.validate_distribution(game):
                    game = random.sample(list(self.stats.all_numbers), LotteryConfig.NUMBERS_PER_GAME)
                X_neg.append(self._generate_features_safe(game))

            # Combina e balanceia
            X = np.vstack((X_pos, np.array(X_neg)))
            y = np.concatenate((y_pos, np.zeros(len(X_neg))))

            # Remove quaisquer NaNs remanescentes
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

            return resample(X, y, n_samples=min(len(X_pos), len(X_neg)) * 2, random_state=42)

        except Exception as e:
            logger.error(f"Erro na preparação dos dados: {e}")
            raise

    def _generate_features(self, numbers: List[int]) -> List[float]:
        """Gera features para um conjunto de números (versão básica)."""
        try:
            return self.stats.get_enhanced_features(numbers)
        except Exception as e:
            logger.warning(f"Erro ao gerar features (versão não segura): {e}")
            return [0.0] * len(self.feature_names)

    def _generate_features_safe(self, numbers: List[int]) -> List[float]:
        """Gera features com tratamento seguro para NaN/inf"""
        try:
            features = self.stats.get_enhanced_features(numbers)
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
            return features
        except Exception as e:
            logger.warning(f"Erro ao gerar features: {e}")
            return [0.0] * len(self.feature_names)

    def predict_proba(self, numbers: List[int]) -> float:
        """
        Predição robusta com tratamento de erros e NaN
        """
        try:
            features = self._generate_features_safe(numbers)
            features = np.array(features).reshape(1, -1)

            if np.isnan(features).any():
                logger.warning("NaN detectado nas features - substituindo por zeros")
                features = np.nan_to_num(features, nan=0.0)

            features_scaled = self.scaler.transform(features)

            probas = []
            for model, weight in zip(self.models, self.model_weights):
                try:
                    if isinstance(model, MLPClassifier) and np.isnan(features_scaled).any():
                        logger.warning("MLP recebeu NaN - usando fallback")
                        proba = 0.5
                    else:
                        proba = model.predict_proba(features_scaled)[0][1]
                    probas.append(proba * weight)
                except Exception as e:
                    logger.error(f"Erro na predição do modelo {type(model).__name__}: {e}")
                    probas.append(0.5 * weight)

            return np.mean(probas)
        except Exception as e:
            logger.error(f"Erro crítico na predição: {e}")
            return 0.5

    def _train_model_safely(self, idx, model, X, y):
        """Wrapper seguro para treinamento com tratamento de NaN"""
        try:
            if np.isnan(X).any() or np.isnan(y).any():
                logger.warning(f"NaN detectado nos dados de treino para modelo {idx + 1}")
                X = np.nan_to_num(X, nan=0.0)
                y = np.nan_to_num(y, nan=0.0)

            model.fit(X, y)
            test_X = np.nan_to_num(X[:1], nan=0.0)
            sample_pred = model.predict(test_X)
            if len(sample_pred) != 1:
                raise ValueError("Predição inválida")
            return (True, model)
        except Exception as e:
            logger.error(f"Erro no treinamento do modelo {idx + 1}: {e}")
            return (False, None)

    def _train_serial(self, X, y):
        """Treinamento serial robusto."""
        for i, model in enumerate(self.models):
            try:
                success, trained_model = self._train_model_safely(i, model, X, y)
                if not success:
                    raise RuntimeError(f"Modelo {i + 1} falhou no treinamento serial")
                self.models[i] = trained_model
            except Exception as e:
                logger.error(f"Falha crítica no treinamento serial: {e}")
                raise

    def _validate_initial_models(self):
        """Validação rigorosa dos modelos."""
        if not self.models:
            raise ValueError("Nenhum modelo disponível para validação")

        test_numbers = [random.sample(list(self.stats.all_numbers), 15) for _ in range(5)]
        X_test = np.array([self._generate_features_safe(nums) for nums in test_numbers])  # Usando a versão segura
        X_test = self.scaler.transform(X_test)

        for i, model in enumerate(self.models):
            try:
                preds = model.predict(X_test)
                if len(preds) != 5:
                    raise ValueError(f"Modelo {i} retornou predições inválidas")

                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(X_test)
                    if probas.shape[0] != 5:
                        raise ValueError(f"Modelo {i} retornou probabilidades inválidas")
            except Exception as e:
                raise RuntimeError(f"Validação falhou para modelo {i}: {e}") from e

    def _emergency_fallback(self, X=None, y=None):
        """Cria modelos mínimos como último recurso."""
        logger.critical("Ativando modo de emergência!")

        try:
            if X is None or y is None:
                X, y = self._prepare_training_data()
                if self.scaler is None:
                    self.scaler = StandardScaler().fit(X)
                X = self.scaler.transform(X)

            self.models = [RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42,
                n_jobs=1
            )]
            self.model_weights = [1.0]

            success, model = self._train_model_safely(0, self.models[0], X, y)
            if not success:
                raise RuntimeError("Falha no modelo de emergência")

            self.models[0] = model
            logger.warning("Modelo de emergência criado (desempenho reduzido)")

        except Exception as e:
            logger.critical("FALHA CATASTRÓFICA: Não foi possível criar modelo mínimo")
            raise RuntimeError("Sistema inoperante") from e

    def _create_mlp(self):
        """Cria MLP com configurações seguras."""
        return MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            early_stopping=True,
            random_state=42,
            max_iter=500,
            batch_size=64,
            learning_rate_init=0.001,
            verbose=False
        )

    def _create_rf(self):
        """Cria RandomForest otimizado."""
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=1,
            verbose=0
        )

    def _create_xgb(self):
        """Cria XGBoost com configurações seguras."""
        return xgb.XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=1,
            verbosity=0
        )

    def save_models(self):
        """Salva modelos no cache com verificação de integridade."""
        try:
            cache_dir = Path(EnhancedLotteryConfig.MODEL_CACHE_DIR)
            cache_dir.mkdir(exist_ok=True, parents=True)

            self._validate_initial_models()

            with open(cache_dir / 'scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)

            for i, model in enumerate(self.models):
                temp_path = cache_dir / f'model_{i}.temp'
                final_path = cache_dir / f'model_{i}.pkl'

                with open(temp_path, 'wb') as f:
                    pickle.dump(model, f)

                with open(temp_path, 'rb') as f:
                    loaded = pickle.load(f)
                    if not hasattr(loaded, 'predict'):
                        raise ValueError(f"Modelo {i} corrompido ao salvar")

                temp_path.replace(final_path)

            logger.info("Modelos salvos no cache com verificação de integridade")
        except Exception as e:
            logger.error(f"Erro ao salvar modelos: {e}")
            for temp_file in cache_dir.glob('*.temp'):
                try:
                    temp_file.unlink()
                except:
                    pass

    @classmethod
    def load_models(cls, stats: AdvancedStatistics) -> Optional['LotteryModelEnsemble']:
        """
        Tenta carregar modelos pré-treinados do cache com verificação robusta.
        """
        try:
            cache_dir = Path(EnhancedLotteryConfig.MODEL_CACHE_DIR)
            if not cache_dir.exists():
                return None

            required_files = ['scaler.pkl'] + [f'model_{i}.pkl' for i in
                                               range(len(EnhancedLotteryConfig.MODEL_WEIGHTS))]
            if not all((cache_dir / f).exists() for f in required_files):
                return None

            ensemble = cls.__new__(cls)
            ensemble.stats = stats

            with open(cache_dir / 'scaler.pkl', 'rb') as f:
                ensemble.scaler = pickle.load(f)

            ensemble.models = []
            for i in range(len(EnhancedLotteryConfig.MODEL_WEIGHTS)):
                with open(cache_dir / f'model_{i}.pkl', 'rb') as f:
                    model = pickle.load(f)
                    if not hasattr(model, 'predict'):
                        raise ValueError(f"Modelo {i} inválido no cache")
                    ensemble.models.append(model)

            ensemble.loaded_from_cache = True
            logger.info("Modelos carregados do cache com sucesso")
            return ensemble

        except Exception as e:
            logger.warning(f"Falha ao carregar do cache: {e}")
            return None

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


def initialize_components(self) -> bool:
    """Inicialização robusta com tratamento completo de erros"""
    try:
        # 1. Verificação dos dados
        if not self.results or len(self.results) < 10:
            logger.error("Dados insuficientes para inicialização")
            return False

        # 2. Cálculo de estatísticas
        try:
            self.stats = AdvancedStatistics(self.results)
        except Exception as e:
            logger.error(f"Falha no cálculo de estatísticas: {str(e)}")
            return False

        # 3. Inicialização do ensemble (com fallback garantido)
        try:
            self.model = LotteryModelEnsemble(self.stats)

            # Verificação básica do ensemble
            if not hasattr(self.model, 'predict_proba') or not self.model.models:
                raise ValueError("Ensemble inválido")

        except Exception as e:
            logger.error(f"Falha crítica nos modelos: {str(e)}")
            return False

        # 4. Inicialização do gerador
        try:
            self.generator = EnhancedLotteryGenerator(self.stats, self.model)
            return True
        except Exception as e:
            logger.error(f"Falha no gerador: {str(e)}")
            return False

    except Exception as e:
        logger.critical(f"Falha na inicialização: {str(e)}")
        return False

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
        games = []
        attempt_count = 0
        max_attempts = num_games * 3  # Aumentado o limite de tentativas

        while len(games) < num_games and attempt_count < max_attempts:
            attempt_count += 1
            game = self._generate_with_model_scores()

            if game and len(set(game)) == LotteryConfig.NUMBERS_PER_GAME:  # Verificação explícita
                games.append(sorted(game))

        # Fallback seguro com verificação reforçada
        if len(games) < num_games:
            additional = self._generate_simple_games(num_games - len(games), games)
            games.extend([g for g in additional if len(set(g)) == LotteryConfig.NUMBERS_PER_GAME])

        return games[:num_games]  # Garante retornar apenas a quantidade solicitada

    def _generate_simple_games(self, quantity: int, existing_games: List[List[int]]) -> List[List[int]]:
        """Geração alternativa garantindo formato correto"""
        simple_games = []

        for _ in range(quantity):
            while True:
                # Gera jogo aleatório básico já como inteiros
                game = sorted(random.sample(range(1, LotteryConfig.TOTAL_NUMBERS + 1),
                                            LotteryConfig.NUMBERS_PER_GAME))

                # Verifica se é único e válido
                if game not in existing_games and game not in simple_games:
                    simple_games.append(game)
                    break

        return simple_games

    def display_games(self, games: List[List[int]]):
        """Exibe os jogos formatados corretamente"""
        print("\nJogos gerados:")
        for i, game in enumerate(games, 1):
            print(f"Jogo {i}: {game}")

    def _generate_fallback_games(self, num_games: int, existing_games: List[List[int]]) -> List[List[int]]:
        """Método alternativo para completar a geração se necessário"""
        fallback_games = []
        for _ in range(num_games):
            while True:
                # Gera jogo aleatório básico
                game = sorted(random.sample(range(1, LotteryConfig.TOTAL_NUMBERS + 1),
                                            LotteryConfig.NUMBERS_PER_GAME))
                game_ints = list(map(int, game))  # Conversão explícita

                # Verifica se é único e válido
                if (game_ints not in existing_games and
                        game_ints not in fallback_games and
                        self._is_valid_basic_game(game_ints)):
                    fallback_games.append(game_ints)
                    break

        return fallback_games

    def _is_valid_basic_game(self, game: List[int]) -> bool:
        """Validação básica reforçada"""
        return (
                len(game) == LotteryConfig.NUMBERS_PER_GAME and
                all(1 <= num <= LotteryConfig.TOTAL_NUMBERS for num in game) and
                len(set(game)) == LotteryConfig.NUMBERS_PER_GAME and  # Sem duplicatas
                GameValidator.validate_distribution(game) and
                GameValidator.avoid_common_patterns(game)
        )

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
    Versão completa e corrigida com todos os métodos implementados.
    """

    def __init__(self, stats: AdvancedStatistics, model: LotteryModelEnsemble):
        """
        Inicializa o gerador avançado com estatísticas e ensemble.

        Parâmetros:
            stats (AdvancedStatistics): Estatísticas avançadas
            model (LotteryModelEnsemble): Ensemble treinado
        """
        super().__init__(stats, model)
        self.stats = stats
        self.ensemble_model = model
        self.base_model = LotteryModel(stats)
        self.generation_attempts = 0
        self.max_optimization_attempts = 500  # Aumentado de 100 para 500
        self.required_advanced_checks = 2  # Reduzido de 3 para 2
        self._cached_scores = {}
        self._last_calculation_time = 0
        self.fallback_count = 0
        self.successful_generations = 0

    def generate_optimized_games(self, num_games: int = LotteryConfig.GAMES_TO_GENERATE) -> List[List[int]]:
        """Gera jogos otimizados garantindo que os números sejam inteiros Python"""
        games = []
        total_attempts = 0

        logger.info(f"Iniciando geração de {num_games} jogos otimizados")

        while len(games) < num_games and total_attempts < self.max_optimization_attempts:
            total_attempts += 1
            try:
                game = self._generate_with_optimization()
                if game:
                    # Conversão explícita para inteiros Python
                    converted_game = [int(num) for num in game]
                    if self._validate_enhanced_game(converted_game, games):
                        games.append(converted_game)
                        self.successful_generations += 1
                        logger.info(f"✅ Jogo {len(games)}/{num_games} gerado (Tentativa {total_attempts})")

            except Exception as e:
                logger.debug(f"Tentativa {total_attempts} falhou: {str(e)}")

        # Fallback controlado se necessário
        if len(games) < num_games:
            missing = num_games - len(games)
            self.fallback_count += missing
            logger.warning(f"⚠️ Gerando {missing} jogos via método básico (fallback)")
            fallback_games = self._generate_smart_fallback_games(missing, games)
            # Conversão explícita para inteiros Python nos jogos de fallback
            games.extend([[int(num) for num in game] for game in fallback_games])

        logger.info(f"🔍 Estatísticas: {self.successful_generations} otimizados, {self.fallback_count} fallbacks")
        return games[:num_games]

    def _generate_with_optimization(self) -> Optional[List[int]]:
        """Geração com múltiplas camadas de otimização"""
        # 1. Geração inicial com modelo
        game = self._generate_with_model_scores()
        if not game:
            return None

        # 2. Balanceamentos progressivos
        game = self._balance_game_progressive(game)

        return sorted(game) if self._is_valid_basic_game(game) else None

    def _generate_with_model_scores(self) -> Optional[List[int]]:
        """Geração de jogos otimizados com garantia de qualidade"""
        try:
            # 1. Pré-processamento: converter para int e filtrar válidos
            all_numbers = list(map(int, self.stats.all_numbers))  # Garante tipo int
            eligible = [num for num in all_numbers if self.stats.delay[num] <= LotteryConfig.MAX_DELAY]

            # 2. Cálculo de scores com penalidades inteligentes
            scores = self._calculate_scores_with_penalties(eligible)

            # 3. Geração do jogo com múltiplas camadas de validação
            game = self._generate_valid_game(eligible, scores)

            return sorted(game) if game else None

        except Exception as e:
            logger.error(f"Erro na geração: {str(e)}", exc_info=True)
            return None

    def _calculate_scores_with_penalties(self, eligible: List[int]) -> Dict[int, float]:
        """Calcula scores com penalidades para números recentes"""
        scores = {}
        recent_games = self.stats.results[-10:] if len(self.stats.results) >= 10 else self.stats.results
        recent_counts = Counter(num for game in recent_games for num in game)

        for num in eligible:
            base_score = self._calculate_number_score(num)
            # Penaliza números que apareceram muito recentemente
            penalty = 0.4 * (recent_counts.get(num, 0) / max(1, len(recent_games))) # Score mínimo garantido
            scores[num] = max(0.1, base_score - penalty)

        return scores

    def _generate_valid_game(self, eligible: List[int], scores: Dict[int, float]) -> List[int]:
        """Gera um jogo válido com até 100 tentativas, garantindo números únicos"""
        for _ in range(100):  # Limite de tentativas
            selected = set()

            # Seleção estratificada garantindo unicidade
            low_nums = [n for n in eligible if n <= 12 and n not in selected]
            high_nums = [n for n in eligible if n > 12 and n not in selected]

            # Seleciona 5 números baixos únicos
            selected.update(self._select_numbers(low_nums, scores, min(5, len(low_nums))))

            # Seleciona 10 números altos únicos
            selected.update(self._select_numbers(high_nums, scores, min(10, len(high_nums))))

            # Verifica se precisa completar o jogo
            needed = LotteryConfig.NUMBERS_PER_GAME - len(selected)
            if needed > 0:
                remaining = [n for n in eligible if n not in selected]
                selected.update(self._select_numbers(remaining, scores, min(needed, len(remaining))))

            game = sorted(selected)
            if (len(game) == LotteryConfig.NUMBERS_PER_GAME and
                    self._is_valid_basic_game(game)):
                return game

        # Fallback: geração aleatória simples com verificação
        return self._generate_fallback_game(eligible)

    def _select_numbers(self, pool: List[int], scores: Dict[int, float], count: int) -> List[int]:
        """Seleciona números ponderados pelo score"""
        if not pool or count <= 0:
            return []

        try:
            weights = np.array([scores.get(num, 0.1) for num in pool])
            weights = np.maximum(weights, 0.01)  # Evita pesos zerados
            weights /= weights.sum()  # Normaliza

            return np.random.choice(pool, size=count, replace=False, p=weights).tolist()
        except:
            # Fallback para seleção uniforme
            return np.random.choice(pool, size=count, replace=False).tolist()

    def _generate_fallback_game(self, eligible: List[int]) -> List[int]:
        """Geração de fallback garantindo números únicos"""
        game = []
        remaining = eligible.copy()

        while len(game) < LotteryConfig.NUMBERS_PER_GAME and remaining:
            # Seleciona aleatoriamente mantendo distribuição básica
            if len(game) < 5:  # Pelo menos 5 números baixos
                candidates = [n for n in remaining if n <= 12]
            else:
                candidates = remaining

            if not candidates:
                candidates = remaining

            num = random.choice(candidates)
            game.append(num)
            remaining.remove(num)  # Remove para evitar duplicatas

        return game if len(game) == LotteryConfig.NUMBERS_PER_GAME else None

    def _normalize_scores(self, scores: Dict[int, float], numbers: List[int]) -> np.ndarray:
        """Normaliza scores para distribuição de probabilidade"""
        weights = np.array([scores.get(num, 0.1) for num in numbers])  # Usa 0.1 como padrão se não existir
        return weights / weights.sum()

    def _balance_game_progressive(self, game: List[int]) -> List[int]:
        """Aplica balanceamentos sem falhar completamente"""
        balanced = game.copy()

        # Ordem de prioridade para balanceamentos
        balance_steps = [
            self._balance_even_odd,
            self._balance_low_high,
            self._balance_quadrants,
            lambda g: self._adjust_prime_count(g, target=5),
            lambda g: self._boost_cycle_numbers(g)
        ]

        for step in balance_steps:
            try:
                result = step(balanced)
                if result:
                    balanced = result
            except:
                continue

        return balanced

    def _validate_enhanced_game(self, game: List[int], existing_games: List[List[int]]) -> bool:
        """Validação mais flexível com menos requisitos obrigatórios"""
        if not game or len(game) != LotteryConfig.NUMBERS_PER_GAME:
            return False

        # Verificações básicas obrigatórias
        basic_checks = [
            GameValidator.validate_distribution(game),
            GameValidator.avoid_common_patterns(game),
            game not in existing_games
        ]
        if not all(basic_checks):
            return False

        # Verificações avançadas (apenas 2 necessárias)
        advanced_checks = [
            self._check_quadrants(game),
            self._check_clusters(game),
            self._check_correlations(game)
        ]

        return sum(advanced_checks) >= self.required_advanced_checks

    def _check_quadrants(self, game: List[int]) -> bool:
        """Versão flexível da verificação de quadrantes"""
        quad_counts = Counter(AdvancedLotteryWheel.get_quadrant(num) for num in game)
        return all(qc >= max(1, EnhancedLotteryConfig.QUADRANT_MIN - 1) for qc in quad_counts.values())

    def _check_clusters(self, game: List[int]) -> bool:
        """Versão flexível de clusters"""
        clusters = {self.stats.number_clusters[num] for num in game}
        return len(clusters) >= 2  # Reduzido de 3 para 2

    def _check_correlations(self, game: List[int]) -> bool:
        """Versão tolerante de correlações"""
        for i, num1 in enumerate(game):
            for num2 in game[i + 1:]:
                if self.stats.number_correlations[num1 - 1][num2 - 1] > 0.8:  # Aumentado de 0.7 para 0.8
                    return False
        return True

    def _generate_smart_fallback_games(self, num_games: int, existing_games: List[List[int]]) -> List[List[int]]:
        """Fallback inteligente que mantém algumas otimizações"""
        fallback_games = []

        while len(fallback_games) < num_games:
            # Tenta gerar com modelo básico primeiro
            game = self._generate_with_base_model()
            if not game:
                # Fallback total se necessário
                game = self._generate_random_valid_game()

            if game and game not in existing_games and game not in fallback_games:
                fallback_games.append(game)

        return fallback_games

    def _generate_with_base_model(self) -> Optional[List[int]]:
        """Versão simplificada usando apenas o modelo básico"""
        try:
            scores = {num: self.base_model.predict_proba([num])
                      for num in self.stats.all_numbers}

            valid_nums = [n for n in scores.keys()
                          if self.stats.delay[n] <= LotteryConfig.MAX_DELAY]

            if not valid_nums:
                return None

            selected = np.random.choice(
                valid_nums,
                size=LotteryConfig.NUMBERS_PER_GAME,
                replace=False,
                p=[scores[num] for num in valid_nums]
            )

            return sorted(selected.tolist())

        except:
            return None

    def _generate_random_valid_game(self) -> List[int]:
        """Geração aleatória com validação básica"""
        while True:
            game = random.sample(list(self.stats.all_numbers), LotteryConfig.NUMBERS_PER_GAME)
            if GameValidator.validate_distribution(game):
                return sorted(game)

    def _balance_even_odd(self, game: List[int]) -> Optional[List[int]]:
        """Balanceia pares e ímpares no jogo"""
        even_count = sum(1 for num in game if num % 2 == 0)

        if even_count < LotteryConfig.MIN_EVEN:
            return self._adjust_number_type(game, target_even=True)
        elif even_count > LotteryConfig.MAX_EVEN:
            return self._adjust_number_type(game, target_even=False)

        return game

    def _balance_low_high(self, game: List[int]) -> Optional[List[int]]:
        """Balanceia números baixos (1-12) e altos (13-25)"""
        low_count = sum(1 for num in game if num <= 12)

        if low_count < LotteryConfig.MIN_LOW:
            return self._adjust_number_range(game, target_low=True)
        elif low_count > LotteryConfig.MAX_LOW:
            return self._adjust_number_range(game, target_low=False)

        return game

    def _balance_quadrants(self, game: List[int]) -> Optional[List[int]]:
        """Garante distribuição mínima por quadrantes"""
        quad_counts = Counter(AdvancedLotteryWheel.get_quadrant(num) for num in game)

        for q in range(1, 5):
            if quad_counts.get(q, 0) < EnhancedLotteryConfig.QUADRANT_MIN:
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

        to_replace = [n for n in game if (n <= threshold) != target_low]
        if not to_replace:
            return game

        replace_num = min(to_replace, key=lambda x: self._calculate_number_score(x))
        new_num = max(candidates, key=lambda x: self._calculate_number_score(x))

        return sorted([n if n != replace_num else new_num for n in game])

    def _adjust_quadrant(self, game: List[int], target_quad: int) -> Optional[List[int]]:
        """Ajusta a distribuição por quadrantes"""
        current_numbers = set(game)
        candidates = [
            n for n in self.stats.all_numbers
            if n not in current_numbers
               and AdvancedLotteryWheel.get_quadrant(n) == target_quad
               and self.stats.delay[n] <= LotteryConfig.MAX_DELAY
        ]

        if not candidates:
            return None

        quad_counts = Counter(AdvancedLotteryWheel.get_quadrant(n) for n in game)
        over_quad = max(quad_counts.keys(), key=lambda x: quad_counts[x])

        to_replace = [n for n in game if AdvancedLotteryWheel.get_quadrant(n) == over_quad]
        if not to_replace:
            return game

        replace_num = min(to_replace, key=lambda x: self._calculate_number_score(x))
        new_num = max(candidates, key=lambda x: self._calculate_number_score(x))

        return sorted([n if n != replace_num else new_num for n in game])

    def _adjust_prime_count(self, game: List[int], target: int = 5) -> Optional[List[int]]:
        """Ajusta a quantidade de números primos no jogo"""
        current_primes = [n for n in game if n in self.stats.prime_numbers]
        current_non_primes = [n for n in game if n not in self.stats.prime_numbers]

        if len(current_primes) < target:
            available_primes = [
                n for n in self.stats.all_numbers
                if n not in game
                   and n in self.stats.prime_numbers
                   and self.stats.delay[n] <= LotteryConfig.MAX_DELAY
            ]

            if not available_primes:
                return None

            game_copy = game.copy()
            for _ in range(target - len(current_primes)):
                if not current_non_primes:
                    break

                worst_num = min(current_non_primes, key=lambda x: self._calculate_number_score(x))
                best_prime = max(available_primes, key=lambda x: self._calculate_number_score(x))

                game_copy.remove(worst_num)
                game_copy.append(best_prime)

                current_non_primes.remove(worst_num)
                available_primes.remove(best_prime)

            return sorted(game_copy)
        else:
            game_copy = game.copy()
            for _ in range(len(current_primes) - target):
                worst_prime = min(current_primes, key=lambda x: self._calculate_number_score(x))
                best_non_prime = max(
                    [n for n in self.stats.all_numbers
                     if n not in game
                     and n not in self.stats.prime_numbers
                     and self.stats.delay[n] <= LotteryConfig.MAX_DELAY],
                    key=lambda x: self._calculate_number_score(x)
                )

                if not best_non_prime:
                    return None

                game_copy.remove(worst_prime)
                game_copy.append(best_non_prime)
                current_primes.remove(worst_prime)

            return sorted(game_copy)

    def _boost_cycle_numbers(self, game: List[int]) -> Optional[List[int]]:
        """
        Aumenta a quantidade de números com ciclos fortes.
        Retorna o jogo ajustado ou None se não for possível.

        Correções aplicadas:
        - Parênteses faltando na chamada de min()
        - Indentação correta do bloco for
        - Verificação de replacements > 0
        """
        current_cycle_strengths = {
            n: self.stats.get_number_cycle_strength(n, len(self.stats.results) + 1)
            for n in game
        }

        strong_cycle_numbers = [
            n for n in self.stats.all_numbers
            if n not in game
               and self.stats.get_number_cycle_strength(n, len(self.stats.results) + 1) > 0.7
               and self.stats.delay[n] <= LotteryConfig.MAX_DELAY
        ]

        if not strong_cycle_numbers:
            return None

        game_copy = game.copy()
        replacements = min(
            3 - sum(1 for s in current_cycle_strengths.values() if s > 0.5),
            len(strong_cycle_numbers)
        )  # Fechamento do parêntese adicionado

        # Verifica se há substituições a fazer
        if replacements <= 0:
            return game_copy

        for _ in range(replacements):
            worst_num = min(game_copy, key=lambda x: current_cycle_strengths.get(x, 0))
            best_cycle_num = max(
                strong_cycle_numbers,
                key=lambda x: self.stats.get_number_cycle_strength(x, len(self.stats.results) + 1)
            )

            game_copy.remove(worst_num)
            game_copy.append(best_cycle_num)
            strong_cycle_numbers.remove(best_cycle_num)

        return sorted(game_copy)

    def _calculate_number_score(self, number: int) -> float:
        """Calcula o score completo de um número individual"""
        if number in self._cached_scores:
            return self._cached_scores[number]

        try:
            ensemble_prob = self.ensemble_model.predict_proba([number])
            delay = self.stats.delay[number]
            freq = self.stats.frequency[number] / max(1, len(self.stats.results))

            if delay <= LotteryConfig.LOW_DELAY_THRESHOLD:
                delay_weight = LotteryConfig.LOW_DELAY_WEIGHT
            elif delay <= LotteryConfig.MEDIUM_DELAY_THRESHOLD:
                delay_weight = LotteryConfig.MEDIUM_DELAY_WEIGHT
            else:
                delay_weight = LotteryConfig.HIGH_DELAY_WEIGHT

            trend = LotteryConfig.TREND_WEIGHT if self.stats.trends[
                                                      number] == "heating" else -LotteryConfig.TREND_WEIGHT

            score = (
                    LotteryConfig.FREQ_WEIGHT * freq +
                    delay_weight * (1 / (delay + 1)) +
                    LotteryConfig.PROB_WEIGHT * ensemble_prob +
                    trend
            )

            self._cached_scores[number] = score
            return score

        except Exception:
            return 0.5  # Valor padrão em caso de erro

    def _is_valid_basic_game(self, game: List[int]) -> bool:
        """Validação básica do jogo incluindo verificação de duplicatas"""
        if len(game) != LotteryConfig.NUMBERS_PER_GAME:
            return False

        # Verifica duplicatas
        if len(set(game)) != len(game):
            return False

        # Restante da validação original...
        numbers_arr = np.array(game)
        even = np.sum(numbers_arr % 2 == 0)
        low = np.sum(numbers_arr <= 12)
        total_sum = np.sum(numbers_arr)

        return all([
            LotteryConfig.MIN_EVEN <= even <= LotteryConfig.MAX_EVEN,
            LotteryConfig.MIN_ODD <= (len(game) - even) <= LotteryConfig.MAX_ODD,
            LotteryConfig.MIN_LOW <= low <= LotteryConfig.MAX_LOW,
            LotteryConfig.MIN_SUM <= total_sum <= LotteryConfig.MAX_SUM,
            GameValidator.avoid_common_patterns(game)
        ])

    def _is_valid_enhanced_game(self, game: List[int], existing_games: List[List[int]]) -> bool:
        """Validação avançada com todas as regras"""
        if not self._is_valid_basic_game(game) or game in existing_games:
            return False

        quadrants = [AdvancedLotteryWheel.get_quadrant(num) for num in game]
        if any(quadrants.count(q) < EnhancedLotteryConfig.QUADRANT_MIN for q in range(1, 5)):
            return False

        clusters = {self.stats.number_clusters[num] for num in game}
        if len(clusters) < 3:
            return False

        return True

    def display_advanced_stats(self, games: List[List[int]]) -> None:
        """Exibe estatísticas detalhadas dos jogos gerados"""
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

    def plot_game(self, game: List[int], filename: str = None):
        """Gera visualização gráfica do jogo no volante"""
        positions = [LotteryWheel.POSITIONS[num] for num in game]
        x, y = zip(*positions)

        plt.figure(figsize=(8, 8))
        plt.scatter(x, y, s=500, c='red', alpha=0.7)

        # Desenha linhas do volante
        for i in range(5):
            plt.axhline(i - 0.5, color='gray', linestyle='--', alpha=0.3)
            plt.axvline(i - 0.5, color='gray', linestyle='--', alpha=0.3)

        # Adiciona números
        for num, (x_pos, y_pos) in LotteryWheel.POSITIONS.items():
            plt.text(x_pos, y_pos, str(num), ha='center', va='center')

        plt.title("Distribuição dos Números no Volante")
        plt.xticks([])
        plt.yticks([])
        plt.xlim(-0.6, 4.6)
        plt.ylim(-0.6, 4.6)

        if filename:
            Path(EnhancedLotteryConfig.PLOT_OUTPUT_DIR).mkdir(exist_ok=True)
            plt.savefig(Path(EnhancedLotteryConfig.PLOT_OUTPUT_DIR) / filename)
        plt.show()


def _boost_cycle_numbers(self, game: List[int]) -> Optional[List[int]]:
    """
    Aumenta a quantidade de números com ciclos fortes.
    Retorna o jogo ajustado ou None se não for possível.
    """
    current_cycle_strengths = {
        n: self.stats.get_number_cycle_strength(n, len(self.stats.results) + 1)
        for n in game
    }

    strong_cycle_numbers = [
        n for n in self.stats.all_numbers
        if n not in game
           and self.stats.get_number_cycle_strength(n, len(self.stats.results) + 1) > 0.7
           and self.stats.delay[n] <= LotteryConfig.MAX_DELAY
    ]

    if not strong_cycle_numbers:
        return None

    game_copy = game.copy()
    replacements = min(
        3 - sum(1 for s in current_cycle_strengths.values() if s > 0.5),
        len(strong_cycle_numbers)
    )

    for _ in range(replacements):
        worst_num = min(game_copy, key=lambda x: current_cycle_strengths.get(x, 0))
        best_cycle_num = max(
            strong_cycle_numbers,
            key=lambda x: self.stats.get_number_cycle_strength(x, len(self.stats.results) + 1)
        )

        game_copy.remove(worst_num)
        game_copy.append(best_cycle_num)
        strong_cycle_numbers.remove(best_cycle_num)

    return sorted(game_copy)


def _calculate_number_score(self, number: int) -> float:
    """
    Calcula o score completo de um número individual.
    """
    if number in self._cached_scores:
        return self._cached_scores[number]

    try:
        ensemble_prob = self.ensemble_model.predict_proba([number])
        delay = self.stats.delay[number]
        freq = self.stats.frequency[number] / max(1, len(self.stats.results))

        if delay <= LotteryConfig.LOW_DELAY_THRESHOLD:
            delay_weight = LotteryConfig.LOW_DELAY_WEIGHT
        elif delay <= LotteryConfig.MEDIUM_DELAY_THRESHOLD:
            delay_weight = LotteryConfig.MEDIUM_DELAY_WEIGHT
        else:
            delay_weight = LotteryConfig.HIGH_DELAY_WEIGHT

        trend = LotteryConfig.TREND_WEIGHT if self.stats.trends[number] == "heating" else -LotteryConfig.TREND_WEIGHT

        score = (
                LotteryConfig.FREQ_WEIGHT * freq +
                delay_weight * (1 / (delay + 1)) +
                LotteryConfig.PROB_WEIGHT * ensemble_prob +
                trend
        )

        self._cached_scores[number] = score
        return score

    except Exception:
        return 0.5  # Valor padrão em caso de erro


def _is_valid_basic_game(self, game: List[int]) -> bool:
    """
    Validação básica do jogo.
    """
    if len(game) != LotteryConfig.NUMBERS_PER_GAME:
        return False

    numbers_arr = np.array(game)
    even = np.sum(numbers_arr % 2 == 0)
    low = np.sum(numbers_arr <= 12)
    total_sum = np.sum(numbers_arr)

    return all([
        LotteryConfig.MIN_EVEN <= even <= LotteryConfig.MAX_EVEN,
        LotteryConfig.MIN_ODD <= (len(game) - even) <= LotteryConfig.MAX_ODD,
        LotteryConfig.MIN_LOW <= low <= LotteryConfig.MAX_LOW,
        LotteryConfig.MIN_SUM <= total_sum <= LotteryConfig.MAX_SUM,
        GameValidator.avoid_common_patterns(game)
    ])


def _is_valid_enhanced_game(self, game: List[int], existing_games: List[List[int]]) -> bool:
    """
    Validação avançada com todas as regras.
    """
    if not self._is_valid_basic_game(game) or game in existing_games:
        return False

    quadrants = [AdvancedLotteryWheel.get_quadrant(num) for num in game]
    if any(quadrants.count(q) < EnhancedLotteryConfig.QUADRANT_MIN for q in range(1, 5)):
        return False

    clusters = {self.stats.number_clusters[num] for num in game}
    if len(clusters) < 3:
        return False

    return True


class LotteryApp:
    """Aplicação principal completa para geração de jogos de loteria."""

    def __init__(self):
        self.results = []
        self.stats = None
        self.model = None
        self.generator = None
        self.file_path = None
        self._initialize_logging()

    def _initialize_logging(self):
        """Configura o sistema de logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('lottery_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self, file_path: str) -> bool:
        """Carrega dados históricos de um arquivo."""
        try:
            self.logger.info(f"Carregando dados do arquivo: {file_path}")

            if not os.path.exists(file_path):
                self.logger.error("Arquivo não encontrado")
                return False

            if os.path.getsize(file_path) == 0:
                self.logger.error("Arquivo vazio")
                return False

            if file_path.lower().endswith(('.xlsx', '.xls')):
                return self._load_excel_data(file_path)
            elif file_path.lower().endswith('.txt'):
                return self._load_text_data(file_path)
            else:
                self.logger.error("Formato de arquivo não suportado")
                return False

        except Exception as e:
            self.logger.error(f"Erro ao carregar dados: {str(e)}", exc_info=True)
            return False

    def _load_excel_data(self, file_path: str) -> bool:
        """Carrega dados de arquivo Excel."""
        try:
            if not self._check_excel_deps():
                self.logger.error("Dependências para Excel não disponíveis")
                return False

            df = pd.read_excel(file_path, header=None, engine='openpyxl')
            valid_games = []

            for idx, row in df.iterrows():
                try:
                    numbers = [int(num) for num in row.dropna()]

                    if len(numbers) != LotteryConfig.NUMBERS_PER_GAME:
                        self.logger.warning(f"Linha {idx + 1} ignorada - quantidade incorreta de números")
                        continue

                    if not all(1 <= num <= LotteryConfig.TOTAL_NUMBERS for num in numbers):
                        invalid = [num for num in numbers if not 1 <= num <= LotteryConfig.TOTAL_NUMBERS]
                        self.logger.warning(f"Linha {idx + 1} ignorada - números inválidos: {invalid}")
                        continue

                    valid_games.append(numbers)

                except ValueError as e:
                    self.logger.warning(f"Linha {idx + 1} ignorada - erro de conversão: {str(e)}")
                    continue

            if not valid_games:
                self.logger.error("Nenhum jogo válido encontrado")
                return False

            self.results = valid_games
            self.file_path = file_path
            self.logger.info(f"Carregados {len(valid_games)} jogos válidos")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao processar Excel: {str(e)}", exc_info=True)
            return False

    def _load_text_data(self, file_path: str) -> bool:
        """Carrega dados de arquivo texto."""
        try:
            valid_games = []

            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        if not line.strip():
                            continue

                        numbers = [int(num) for num in line.strip().replace(',', ' ').split()]

                        if len(numbers) != LotteryConfig.NUMBERS_PER_GAME:
                            self.logger.warning(f"Linha {line_num} ignorada - quantidade incorreta de números")
                            continue

                        if not all(1 <= num <= LotteryConfig.TOTAL_NUMBERS for num in numbers):
                            invalid = [num for num in numbers if not 1 <= num <= LotteryConfig.TOTAL_NUMBERS]
                            self.logger.warning(f"Linha {line_num} ignorada - números inválidos: {invalid}")
                            continue

                        valid_games.append(numbers)

                    except ValueError as e:
                        self.logger.warning(f"Linha {line_num} ignorada - erro de conversão: {str(e)}")
                        continue

            if not valid_games:
                self.logger.error("Nenhum jogo válido encontrado")
                return False

            self.results = valid_games
            self.file_path = file_path
            self.logger.info(f"Carregados {len(valid_games)} jogos válidos")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao processar arquivo texto: {str(e)}", exc_info=True)
            return False

    def _check_excel_deps(self) -> bool:
        """Verifica dependências para Excel."""
        try:
            import openpyxl
            return True
        except ImportError:
            try:
                import subprocess
                subprocess.check_call(['pip', 'install', 'openpyxl'])
                return True
            except:
                return False

    def initialize_components(self) -> bool:
        """Inicializa todos os componentes do sistema."""
        try:
            min_required = max(15, LotteryConfig.TREND_WINDOW_SIZE * 2)
            if len(self.results) < min_required:
                self.logger.error(f"Dados insuficientes. Mínimo requerido: {min_required} jogos")
                return False

            self.logger.info("Calculando estatísticas...")
            try:
                self.stats = AdvancedStatistics(self.results)
                self.logger.info("Estatísticas calculadas com sucesso")
            except Exception as e:
                self.logger.error(f"Falha no cálculo de estatísticas: {str(e)}", exc_info=True)
                return False

            self.logger.info("Inicializando modelos...")
            try:
                self.model = LotteryModelEnsemble(self.stats)
                self.logger.info("Modelos inicializados com sucesso")
            except Exception as e:
                self.logger.error(f"Falha na inicialização dos modelos: {str(e)}", exc_info=True)
                return False

            self.logger.info("Inicializando gerador...")
            try:
                self.generator = EnhancedLotteryGenerator(self.stats, self.model)
                self.logger.info("Componentes inicializados com sucesso")
                return True
            except Exception as e:
                self.logger.error(f"Falha no gerador: {str(e)}", exc_info=True)
                return False

        except Exception as e:
            self.logger.critical(f"Falha na inicialização: {str(e)}", exc_info=True)
            return False

    def generate_games(self, num_games: int = None) -> List[List[int]]:
        """Gera jogos otimizados com fallback controlado."""
        num_games = num_games or LotteryConfig.GAMES_TO_GENERATE

        if not all([self.stats, self.model, self.generator]):
            self.logger.error("Componentes não inicializados corretamente")
            return []

        try:
            self.logger.info(f"Iniciando geração de {num_games} jogos")
            games = self.generator.generate_optimized_games(num_games)

            if len(games) < num_games:
                self.logger.warning(f"Apenas {len(games)}/{num_games} jogos puderam ser gerados")

            return games

        except Exception as e:
            self.logger.error(f"Erro ao gerar jogos: {str(e)}", exc_info=True)
            return []

    def display_games(self, games: List[List[int]]) -> None:
        """Exibe os jogos formatados corretamente como inteiros simples"""
        if not games:
            print("\nNenhum jogo foi gerado.")
            return

        print("\nJogos gerados:")
        for i, game in enumerate(games, 1):
            # Garante que todos os números são inteiros Python
            converted_game = [int(num) for num in game]
            print(f"Jogo {i}: {sorted(converted_game)}")

    def save_games(self, games: List[List[int]], filename: str = "jogos_gerados.txt") -> bool:
        """Salva os jogos garantindo que todos os números são inteiros Python"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for game in games:
                    # Garante conversão para inteiros Python
                    converted_game = [int(num) for num in game]
                    f.write(" ".join(map(str, sorted(converted_game))) + "\n")
            self.logger.info(f"Jogos salvos em {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao salvar jogos: {str(e)}", exc_info=True)
            return False

    def display_advanced_stats(self, games: List[List[int]]) -> None:
        """Exibe estatísticas avançadas com conversão de tipos"""
        if not games:
            print("Nenhum jogo para exibir estatísticas")
            return

        print("\nEstatísticas Avançadas:")
        print("{:<10} {:<8} {:<8} {:<8} {:<10}".format(
            "Jogo", "Pares", "Ímpares", "Baixos", "Soma"))

        for i, game in enumerate(games, 1):
            # Garante conversão para inteiros Python
            numbers = np.array([int(num) for num in game])
            even = np.sum(numbers % 2 == 0)
            low = np.sum(numbers <= 12)
            total = np.sum(numbers)

            print("{:<10} {:<8} {:<8} {:<8} {:<10}".format(
                f"Jogo {i}", even, len(game) - even, low, total))

    def display_advanced_stats(self, games: List[List[int]]) -> None:
        """Exibe estatísticas avançadas dos jogos."""
        if not games:
            print("Nenhum jogo para exibir estatísticas")
            return

        print("\nEstatísticas Avançadas:")
        print("{:<10} {:<8} {:<8} {:<8} {:<10}".format(
            "Jogo", "Pares", "Ímpares", "Baixos", "Soma"))

        for i, game in enumerate(games, 1):
            numbers = np.array(game)
            even = np.sum(numbers % 2 == 0)
            low = np.sum(numbers <= 12)
            total = np.sum(numbers)

            print("{:<10} {:<8} {:<8} {:<8} {:<10}".format(
                f"Jogo {i}", even, len(game) - even, low, total))

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


def save_games(self, games: List[List[int]], filename: str = "jogos_gerados.txt"):
    """Salva os jogos gerados em arquivo"""
    with open(filename, 'w') as f:
        for game in games:
            f.write(" ".join(map(str, sorted(game))) + "\n")
    logger.info(f"Jogos salvos em {filename}")


def display_advanced_stats(self, games: List[List[int]]):
    """Exibe estatísticas avançadas com visualização"""
    super().display_advanced_stats(games)

    # Gera gráficos para os primeiros 3 jogos
    for i, game in enumerate(games[:3], 1):
        self.generator.plot_game(game, f"jogo_{i}.png")


def main():
    """Função principal com correções para solicitar o arquivo"""
    print("=== SISTEMA AVANÇADO DE GERAÇÃO DE JOGOS DE LOTERIA v2.0 ===")
    print("=== Desenvolvido por Natanael ===\n")

    try:
        # Verifica dependências
        check_dependencies()

        app = LotteryApp()

        # 1. Solicitação do arquivo de forma explícita
        print("\n[1/4] Selecione o arquivo com resultados históricos:")
        print("Por favor, escolha um arquivo Excel (.xlsx) ou texto (.txt) contendo os resultados anteriores.")

        while True:
            file_path = FileHandler.select_file()
            if file_path:
                break
            print("\nNenhum arquivo selecionado. Por favor, selecione um arquivo válido.")

        # 2. Carregamento dos dados
        print(f"\n[2/4] Carregando dados de: {file_path}")
        if not app.load_data(file_path):
            print("\nERRO: Não foi possível carregar os dados. Verifique o formato do arquivo.")
            input("Pressione Enter para sair...")
            return

        # 3. Inicialização do sistema
        print("\n[3/4] Inicializando componentes...")
        if not app.initialize_components():
            print("\nERRO: Falha ao inicializar o sistema.")
            input("Pressione Enter para sair...")
            return

        # 4. Geração de jogos
        print("\n[4/4] Gerando jogos otimizados")
        while True:
            try:
                num_games = input("\nQuantos jogos deseja gerar? (Padrão=7, Máximo=50): ").strip()
                num_games = int(num_games) if num_games else LotteryConfig.GAMES_TO_GENERATE
                if 1 <= num_games <= 50:
                    break
                print("Por favor, digite um número entre 1 e 50.")
            except ValueError:
                print("Entrada inválida. Usando valor padrão (7).")
                num_games = LotteryConfig.GAMES_TO_GENERATE
                break

        games = app.generate_games(num_games)
        app.display_games(games)

        # Opções adicionais
        print("\nDeseja ver estatísticas detalhadas? (S/N)")
        if input().strip().lower() == 's':
            app.display_advanced_stats(games)

        print("\nDeseja salvar os jogos gerados? (S/N)")
        if input().strip().lower() == 's':
            filename = input("Nome do arquivo (deixe em branco para 'jogos_gerados.txt'): ").strip()
            filename = filename or "jogos_gerados.txt"
            app.save_games(games, filename)
            print(f"Jogos salvos em {filename}")

    except Exception as e:
        print(f"\nERRO INESPERADO: {str(e)}")
    finally:
        print("\nProcesso concluído.")
        input("Pressione Enter para sair...")


if __name__ == "__main__":
    main()
