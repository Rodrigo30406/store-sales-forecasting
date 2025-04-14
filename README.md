# 🛒 Store Sales - Time Series Forecasting (Kaggle Competition)

Este repositorio contiene la solución desarrollada para la competencia [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview) organizada por Kaggle y Corporación Favorita.

## 📌 Objetivo

Predecir las ventas diarias de productos para una cadena de supermercados con múltiples tiendas y familias de productos, utilizando series de tiempo históricas y otras variables contextuales.

## 📁 Estructura del repositorio

```
store-sales-forecasting/
├── data/                  # Datos locales (no versionados por Git)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/             # Notebooks de experimentación y EDA
│   ├── 01_eda.ipynb
│   ├── 02_model_lgbm.ipynb
│   └── 03_final_submission.ipynb
├── src/                   # Código modularizado
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── model.py
│   └── utils.py
├── submissions/           # Archivos para enviar a Kaggle
│   └── submission.csv
├── requirements.txt
└── README.md
```

## 🧪 Tecnologías utilizadas

- Python 3.x
- Pandas, NumPy
- LightGBM
- Scikit-learn
- Matplotlib / Seaborn
- (opcional) Prophet, XGBoost, CatBoost

## 🧠 Enfoque de la solución

1. **EDA**: análisis de patrones de ventas por tienda, producto y fecha.
2. **Feature engineering**:
   - Variables temporales: día, mes, año, día de la semana, feriados.
   - Lag features y rolling averages.
   - Encoding de tiendas y familias.
3. **Modelado**: uso de `LightGBM` para predecir log-transformed sales.
4. **Validación**: backtesting sobre los últimos 6 días.
5. **Predicción final**: generación del `submission.csv` para Kaggle.

## 🚀 Cómo ejecutar

1. Clonar el repositorio:

```bash
git clone https://github.com/tu_usuario/store-sales-forecasting.git
cd store-sales-forecasting
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Descargar el dataset desde Kaggle y colocarlo en la carpeta `data/`.

4. Ejecutar los notebooks de `notebooks/` en orden para entrenamiento y predicción.

## 📈 Métricas

La métrica oficial de la competencia es **RMSLE (Root Mean Squared Logarithmic Error)**.

## 📋 Referencias

- [Página oficial de la competencia](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview)
- [Documentación de LightGBM](https://lightgbm.readthedocs.io/)
- [TFDS Ecuador holidays](https://www.kaggle.com/datasets/ferbaquero/holidays-in-ecuador)

## 🤝 Autor

Orasio — *Solución desarrollada para la competencia Kaggle Store Sales Forecasting 2025*