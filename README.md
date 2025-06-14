# 🎯 Supesequência Loto – Gerador Avançado de Jogos da Lotofácil

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/version-1.0.0-informational)]()

> Um sistema completo e inteligente para geração de jogos otimizados da Lotofácil, combinando estatística, inteligência artificial e validação geométrica.

---

## 🚀 Visão Geral

O **Supesequência Loto** é um sistema de geração de jogos da Lotofácil que utiliza:

- 📊 **Análise estatística** de resultados históricos  
- 🧠 **Modelos de machine learning** (MLP, Random Forest, XGBoost – ensemble)  
- 🎯 **Validação probabilística e geométrica no volante**
- 📐 **Balanceamento por quadrantes e clusters de números**
- ♻️ **Otimização multi-nível** com fallback inteligente

---

## ⚙️ Funcionalidades

- Importação de resultados históricos (Excel ou .txt)
- Cálculo de frequência, atraso e tendência dos números
- Geração de jogos com balanceamento par/ímpar e alto/baixo
- Previsão de probabilidade de sucesso via modelos de ML
- Validação com regras avançadas: clusters, quadrantes e padrões
- Interface simples via console e integração com GUI para arquivos

---

## 🧠 Tecnologias e Bibliotecas

- Python 3.8+
- `NumPy`, `Pandas`, `scikit-learn`, `XGBoost`
- `tkinter` (para seleção de arquivos)
- `SciPy` (análise geométrica)
- `OpenPyXL`, `xlrd` (para suporte a Excel)

---

## 📦 Instalação

```bash
git clone https://github.com/seu-usuario/supesequencia-loto.git
cd supesequencia-loto
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows
pip install -r requirements.txt
```

> Use `python supesequencia_loto_facil_1_0.py` para iniciar o sistema.

---

## 📂 Estrutura

```bash
supesequencia-loto/
├── supesequencia_loto_facil_1_0.py
├── resultados_exemplo.xlsx
├── README.md
└── requirements.txt
```

---

## 🧪 Como Usar

1. Execute o script principal:
```bash
python supesequencia_loto_facil_1_0.py
```

2. Selecione o arquivo de resultados históricos (.xlsx ou .txt)

3. O sistema irá:
   - Treinar os modelos
   - Gerar jogos otimizados com base em estatísticas e IA
   - Exibir as sequências geradas

4. Um arquivo de exemplo será criado se nenhum for selecionado.

---

## 📊 Exemplo de Saída

```
Jogos gerados:
Sequencia 1: [  2,  3,  5,  7,  9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 25 ]
Sequencia 2: [  2,  3,  5,  6,  7,  9, 10, 11, 15, 16, 18, 20, 21, 23, 25 ]
...
```

---

## 🧑‍💻 Autor

Desenvolvido por **Natanael Silva**  
💬 Contato: **rnh.persnalizados@gmail.com**  
📍 Brasil

---

## 📄 Licença

Este projeto está licenciado sob os termos da licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
