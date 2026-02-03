import os
from typing import Dict, Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = "healthcare-dataset-stroke-data.csv"


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Lê o arquivo de dados (XLS ou CSV) de stroke.
    """
    print(f"Carregando dataset de {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de dados não encontrado em: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    else:
        # pandas usa engines diferentes para .xls/.xlsx.
        # Certifique-se de ter xlrd (<2.0.0) instalado para .xls.
        df = pd.read_excel(path)

    return df


def build_model(df: pd.DataFrame) -> Pipeline:
    """
    Cria e treina um modelo simples de classificação de AVC
    usando regressão logística + pré-processamento.
    """
    # Coluna-alvo padrão deste dataset do Kaggle
    target_col = "stroke"
    if target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{target_col}' não encontrada no dataset.")

    # Remove identificadores que não ajudam na predição
    drop_cols = [c for c in ["id"] if c in df.columns]
    X = df.drop(columns=[target_col] + drop_cols)
    y = df[target_col]

    # Definição explícita das colunas esperadas no dataset
    categorical_cols = [
        col
        for col in [
            "gender",
            "ever_married",
            "work_type",
            "Residence_type",
            "smoking_status",
        ]
        if col in X.columns
    ]

    numeric_cols = [
        col
        for col in [
            "age",
            "hypertension",
            "heart_disease",
            "avg_glucose_level",
            "bmi",
        ]
        if col in X.columns
    ]

    # Verificações simples para evitar erros silenciosos
    if not categorical_cols and not numeric_cols:
        raise ValueError("Nenhuma coluna de entrada reconhecida para o modelo.")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    model.fit(X, y)
    return model


def ask_choice(prompt: str, options_map: Dict[str, Any], descriptions: Dict[str, str] = None) -> Any:
    """
    Pergunta ao usuário para escolher uma opção.
    options_map: mapeia a resposta digitada -> valor usado no modelo.
    descriptions: mapeia a resposta digitada -> descrição legível (opcional).
    """
    if descriptions:
        options_str = " / ".join([f"{k} ({descriptions[k]})" for k in options_map.keys()])
    else:
        options_str = " / ".join(options_map.keys())
    
    while True:
        raw = input(f"{prompt} [{options_str}]: ").strip().lower()
        if raw in options_map:
            return options_map[raw]
        print(f"Por favor responda com uma das opções: {options_str}")


def ask_float(prompt: str, allow_empty: bool = False, min_val: float | None = None, allow_unknown: bool = False) -> float | None:
    """
    Pergunta um número float ao usuário.
    allow_unknown: se True, aceita "não sei", "nao sei", "não sei", etc. e retorna None.
    """
    while True:
        raw = input(f"{prompt}: ").strip().replace(",", ".")
        if allow_unknown and raw.lower() in ("não sei", "nao sei", "n sei", "não sei", "ns", "?"):
            return None
        if allow_empty and raw == "":
            return None
        try:
            value = float(raw)
            if min_val is not None and value < min_val:
                print(f"Valor mínimo permitido é {min_val}.")
                continue
            return value
        except ValueError:
            if allow_unknown:
                print("Por favor, informe um número válido ou digite 'não sei' se não souber.")
            else:
                print("Por favor, informe um número válido.")


def ask_yes_no(prompt: str) -> bool:
    """
    Retorna True para SIM, False para NÃO.
    """
    while True:
        raw = input(f"{prompt} [s/n]: ").strip().lower()
        if raw in ("s", "sim", "y", "yes"):
            return True
        if raw in ("n", "nao", "não", "no"):
            return False
        print("Responda com 's' para sim ou 'n' para não.")


def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """
    Calcula o BMI (Índice de Massa Corporal).
    BMI = peso (kg) / altura (m)²
    """
    return weight_kg / (height_m ** 2)


def collect_user_answers() -> Dict[str, Any]:
    """
    Faz perguntas em português, retornando um dicionário
    no formato esperado pelo modelo de stroke.
    """
    print("\n=== Calculadora de Risco de AVC (Stroke) ===\n")
    print("Responda às perguntas abaixo. Nenhuma informação será armazenada.\n")

    gender = ask_choice(
        "Sexo",
        {
            "m": "Male",
            "f": "Female",
            "o": "Other",
        },
        descriptions={
            "m": "Masculino",
            "f": "Feminino",
            "o": "Outro",
        },
    )

    age = ask_float("Idade em anos (ex: 45)", min_val=0)

    hypertension = ask_yes_no("Você tem diagnóstico de hipertensão (pressão alta)?")
    heart_disease = ask_yes_no("Você tem histórico de doença cardíaca?")

    ever_married = ask_yes_no("Você já foi casado(a)?")

    work_type = ask_choice(
        "Tipo de trabalho",
        {
            "c": "children",
            "g": "Govt_job",
            "n": "Never_worked",
            "p": "Private",
            "s": "Self-employed",
        },
        descriptions={
            "c": "Criança",
            "g": "Trabalho público",
            "n": "Nunca trabalhou",
            "p": "Setor privado",
            "s": "Autônomo",
        },
    )

    residence_type = ask_choice(
        "Tipo de residência",
        {
            "u": "Urban",
            "r": "Rural",
        },
        descriptions={
            "u": "Urbana",
            "r": "Rural",
        },
    )

    avg_glucose_level = ask_float(
        "Nível médio de glicose no sangue (mg/dL, ex: 105.3). Digite 'não sei' se não souber",
        min_val=0,
        allow_unknown=True,
    )

    # Pergunta altura e peso para calcular BMI
    print("\nVamos calcular seu BMI (Índice de Massa Corporal):")
    height_cm = ask_float("Altura em centímetros (ex: 175)", min_val=0)
    weight_kg = ask_float("Peso em quilogramas (ex: 70)", min_val=0)
    bmi = calculate_bmi(weight_kg, height_cm / 100)  # Converte cm para metros
    print(f"Seu BMI calculado: {bmi:.1f}\n")

    smoking_status = ask_choice(
        "Status de tabagismo",
        {
            "n": "never smoked",
            "f": "formerly smoked",
            "s": "smokes",
            "u": "Unknown",
        },
        descriptions={
            "n": "Nunca fumou",
            "f": "Ex-fumante",
            "s": "Fuma atualmente",
            "u": "Desconhecido",
        },
    )

    # Monta o dicionário no formato das colunas do dataset
    sample = {
        "gender": gender,
        "age": age,
        "hypertension": 1 if hypertension else 0,
        "heart_disease": 1 if heart_disease else 0,
        "ever_married": "Yes" if ever_married else "No",
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status,
    }

    return sample


def main() -> None:
    # Carrega os dados e treina o modelo rapidamente
    print("Carregando dataset e treinando modelo... (isso pode levar alguns segundos)")
    df = load_dataset(DATA_PATH)
    model = build_model(df)

    # Coleta respostas do usuário
    user_sample = collect_user_answers()
    sample_df = pd.DataFrame([user_sample])

    # Faz previsão de probabilidade
    proba = model.predict_proba(sample_df)[0, 1]

    print("\n=== Resultado ===")
    print(f"Probabilidade estimada de AVC: {proba * 100:.1f}%")
    if proba >= 0.5:
        print("⚠️  Risco relativamente ALTO segundo o modelo (não é diagnóstico médico).")
    else:
        print("Risco relativamente BAIXO segundo o modelo (não é diagnóstico médico).")
    print("\nEste resultado é apenas educativo e NÃO substitui avaliação médica.\n")


if __name__ == "__main__":
    main()


