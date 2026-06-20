# NFL Game Momentum Model 🏈

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Prophet](https://img.shields.io/badge/Prophet-1.1.5-orange.svg)](https://facebook.github.io/prophet/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00.svg)](https://www.tensorflow.org/)

**Modelo de Series de Tiempo para Predicción de Momentum en Partidos de NFL**

Un proyecto de análisis predictivo que utiliza múltiples modelos de series de tiempo (Prophet, ARIMA/SARIMAX, y LSTM) para predecir el momentum del juego en partidos de la NFL basándose en puntos, posesiones, sacks y turnovers.

---

## 👤 Autor

**César Adrián Delgado Díaz**

- 🌐 Portfolio: [tu-portfolio.com](https://tu-portfolio.com)
- 💼 LinkedIn: [linkedin.com/in/cesar-delgado-diaz](https://www.linkedin.com/in/cesar-delgado-diaz)
- 💻 GitHub: [github.com/cesar530](https://github.com/cesar530)

---

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema de análisis predictivo para el momentum de partidos de NFL utilizando técnicas avanzadas de machine learning y deep learning. El modelo analiza la dinámica del juego en tiempo real y predice cambios en el momentum basándose en múltiples características del partido.

### 🎯 Objetivos

- **Predicción de Momentum**: Modelar y predecir los cambios en el momentum del juego
- **Análisis Comparativo**: Comparar la efectividad de múltiples técnicas de modelado
- **Visualización Interactiva**: Proporcionar insights visuales sobre la progresión del juego
- **Demostración de Habilidades**: Showcase de competencias en análisis de series de tiempo y ML

### 🔑 Características Principales

- ✅ Implementación de **tres modelos** de series de tiempo:
  - **Prophet** (Facebook's forecasting tool)
  - **ARIMA/SARIMAX** (modelo estadístico clásico)
  - **LSTM** (Long Short-Term Memory neural networks)
- ✅ Ingeniería de características avanzada para momentum
- ✅ Visualizaciones interactivas con Plotly
- ✅ Análisis comparativo de performance de modelos
- ✅ Generación de datos sintéticos realistas
- ✅ Sistema modular y reutilizable

---

## 🛠️ Tecnologías Utilizadas

### Core ML/DL
- **Prophet 1.1.5** - Forecasting
- **statsmodels 0.14.0** - ARIMA/SARIMAX
- **TensorFlow 2.15.0** - Deep Learning (LSTM)
- **Keras 2.15.0** - Neural Networks API

### Data Science Stack
- **NumPy 1.24.3** - Computación numérica
- **Pandas 2.0.3** - Manipulación de datos
- **Scikit-learn 1.3.2** - Preprocessing y métricas

### Visualización
- **Matplotlib 3.7.2** - Gráficos estáticos
- **Seaborn 0.12.2** - Visualización estadística
- **Plotly 5.17.0** - Gráficos interactivos

### Entorno
- **Jupyter Notebook 7.0.6** - Desarrollo interactivo
- **Python 3.10+** - Lenguaje base

---

## 📦 Instalación

### Requisitos Previos

- Python 3.10 o superior
- pip (gestor de paquetes de Python)
- Entorno virtual (recomendado)

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/cesar530/07_NFL_Game_Momentum.git
cd 07_NFL_Game_Momentum
```

### Paso 2: Crear Entorno Virtual

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/Mac
```bash
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Nota**: Las versiones en `requirements.txt` están específicamente seleccionadas para evitar el error `ValueError: numpy.dtype size changed`.

---

## 🚀 Uso

### Opción 1: Jupyter Notebook (Recomendado)

1. Inicia Jupyter Notebook:
```bash
jupyter notebook
```

2. Abre el archivo `NFL_Game_Momentum_Analysis.ipynb`

3. Ejecuta las celdas secuencialmente

### Opción 2: Python Scripts

```python
from game_momentum_model import GameMomentumModel
from utils import generate_synthetic_game_data, calculate_momentum_features

# Generar datos
game_data = generate_synthetic_game_data(n_plays=150)
game_data = calculate_momentum_features(game_data)

# Preparar datos
X = game_data[feature_columns]
y = game_data['momentum_score']

# Entrenar modelo
model = GameMomentumModel('lstm')
model.fit(X, y, epochs=50, batch_size=16)

# Predecir
predictions = model.predict(X_test)
```

### Opción 3: Entrenar Todos los Modelos

```python
from game_momentum_model import train_all_models

models = train_all_models(X_train, y_train, X_test, verbose=True)
```

---

## 📊 Estructura del Proyecto

```
07_NFL_Game_Momentum/
│
├── NFL_Game_Momentum_Analysis.ipynb  # Notebook principal con análisis completo
├── game_momentum_model.py            # Clases de modelos (Prophet, ARIMA, LSTM)
├── utils.py                          # Funciones auxiliares y visualización
├── requirements.txt                  # Dependencias del proyecto
├── .gitignore                        # Archivos ignorados por Git
├── LICENSE                           # Licencia MIT
└── README.md                         # Este archivo
```

---

## 📈 Modelos Implementados

### 1. Prophet
- **Ventajas**: Manejo robusto de tendencias y estacionalidad
- **Uso**: Forecasting a corto y mediano plazo
- **Configuración**: Regressores personalizados para features del juego

### 2. ARIMA/SARIMAX
- **Ventajas**: Modelo estadístico interpretable
- **Uso**: Análisis de series de tiempo tradicionales
- **Configuración**: Orden (p, d, q) = (2, 1, 2) con variables exógenas

### 3. LSTM
- **Ventajas**: Captura dependencias temporales complejas
- **Uso**: Predicción de patrones no lineales
- **Arquitectura**: 
  - 2 capas LSTM (64 y 32 unidades)
  - Dropout layers (0.2)
  - Dense layers para output

---

## 📉 Métricas de Evaluación

El proyecto utiliza múltiples métricas para evaluación comprehensiva:

- **RMSE** (Root Mean Squared Error) - Error cuadrático medio
- **MAE** (Mean Absolute Error) - Error absoluto medio
- **R²** (R-squared) - Coeficiente de determinación
- **MAPE** (Mean Absolute Percentage Error) - Error porcentual absoluto medio

---

## 🔍 Features del Modelo

### Features de Input

1. **Score Differential** - Diferencia en puntos entre equipos
2. **Total Score** - Puntos totales del juego
3. **Field Position** - Posición en el campo (yardas)
4. **Home Possession** - Indicador de posesión (home/away)
5. **Game Time Elapsed** - Tiempo transcurrido del juego
6. **Home Score Momentum** - Momentum de puntos del equipo local
7. **Away Score Momentum** - Momentum de puntos del visitante
8. **Possession Momentum** - Momentum de cambios de posesión
9. **Field Position Momentum** - Momentum de posición en el campo

### Target Variable

- **Momentum Score** - Indicador compuesto de momentum del juego

---

## 📊 Visualizaciones Incluidas

1. **Game Progression Plot** - Progresión interactiva del partido (Plotly)
2. **Momentum Heatmap** - Mapa de calor del momentum por quarter
3. **Correlation Matrix** - Matriz de correlación entre features
4. **Model Predictions Comparison** - Comparación visual de predicciones
5. **Performance Metrics Bars** - Gráficos de barras de métricas
6. **Residuals Analysis** - Análisis de residuales por modelo
7. **LSTM Training History** - Curvas de loss y MAE

---

## 🎓 Habilidades Demostradas

Este proyecto demuestra competencias en:

- ✅ **Análisis de Series de Tiempo** - Modelado temporal avanzado
- ✅ **Machine Learning** - Múltiples algoritmos (Prophet, ARIMA)
- ✅ **Deep Learning** - Redes neuronales recurrentes (LSTM)
- ✅ **Feature Engineering** - Creación de características de momentum
- ✅ **Data Visualization** - Gráficos estáticos e interactivos
- ✅ **Python Programming** - Código modular y bien documentado
- ✅ **Model Evaluation** - Análisis comparativo robusto
- ✅ **Statistical Analysis** - Interpretación de métricas

---

## 🔧 Configuración Avanzada

### Personalizar Modelos

#### Prophet
```python
model = GameMomentumModel('prophet')
model.fit(
    X_train, 
    y_train,
    changepoint_prior_scale=0.1,  # Sensibilidad a cambios de tendencia
    seasonality_prior_scale=10,   # Fuerza de estacionalidad
    seasonality_mode='additive'   # Tipo de estacionalidad
)
```

#### ARIMA
```python
model = GameMomentumModel('arima')
model.fit(
    X_train,
    y_train,
    order=(p, d, q)  # p: autoregresivo, d: diferenciación, q: media móvil
)
```

#### LSTM
```python
model = GameMomentumModel('lstm')
model.fit(
    X_train,
    y_train,
    sequence_length=10,    # Ventana de tiempo
    epochs=50,             # Iteraciones
    batch_size=16,         # Tamaño de batch
    lstm_units=64,         # Neuronas primera capa
    lstm_units_2=32,       # Neuronas segunda capa
    dropout=0.2,           # Tasa de dropout
    learning_rate=0.001    # Tasa de aprendizaje
)
```

---

## 🐛 Solución de Problemas

### Error: numpy.dtype size changed

**Solución**: Las versiones en `requirements.txt` están configuradas para evitar este error. Asegúrate de instalar las versiones exactas:

```bash
pip install --force-reinstall -r requirements.txt
```

### Error: Module not found

**Solución**: Verifica que el entorno virtual esté activado y todas las dependencias instaladas:

```bash
pip list
pip install -r requirements.txt
```

### LSTM no converge

**Solución**: Ajusta los hiperparámetros:
- Aumenta el número de epochs
- Reduce el learning rate
- Aumenta el tamaño del batch
- Agrega más datos de entrenamiento

---

## 🚀 Mejoras Futuras

- [ ] Integración con API de datos reales de NFL
- [ ] Implementación de ensemble models
- [ ] Optimización de hiperparámetros con GridSearch/Optuna
- [ ] Deployment en la nube (AWS/Azure/GCP)
- [ ] Dashboard interactivo con Streamlit/Dash
- [ ] Incorporar más features (clima, injuries, home advantage)
- [ ] Sistema de alertas en tiempo real
- [ ] Análisis de múltiples partidos simultáneos

---

## 📚 Referencias

### Documentación Técnica
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [statsmodels SARIMAX](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
- [TensorFlow LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)

### Papers y Recursos
- Time Series Forecasting: Principles and Practice (Hyndman & Athanasopoulos)
- LSTM Networks for Time Series Prediction
- NFL Analytics Research

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

```
MIT License

Copyright (c) 2026 César Adrián Delgado Díaz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📞 Contacto

**César Adrián Delgado Díaz**

- 📧 Email: [Disponible en LinkedIn](https://www.linkedin.com/in/cesar-delgado-diaz)
- 💼 LinkedIn: [linkedin.com/in/cesar-delgado-diaz](https://www.linkedin.com/in/cesar-delgado-diaz)
- 🌐 Portfolio: [tu-portfolio.com](https://tu-portfolio.com)
- 💻 GitHub: [github.com/cesar530](https://github.com/cesar530)

---

## ⭐ Agradecimientos

- Comunidad de Data Science en español
- Desarrolladores de Prophet, statsmodels y TensorFlow
- NFL por los datos públicos disponibles

---

<div align="center">

**Si este proyecto te resultó útil, considera darle una ⭐ en GitHub!**

Made with ❤️ by César Adrián Delgado Díaz

</div>
