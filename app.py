import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
import io
from pathlib import Path
import joblib
from model_pipeline import create_and_save_pipeline, load_pipeline
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Настройка страницы
st.set_page_config(
    page_title="Анализ цен на автомобили",
    page_icon="🚗",
    layout="wide"
)

# Инициализация состояния сессии
if 'df_train' not in st.session_state:
    st.session_state.df_train = None
if 'df_test' not in st.session_state:
    st.session_state.df_test = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# Заголовок приложения
st.title("🚗 Аналитическая система прогнозирования цен на автомобили")
st.markdown("---")

# Боковая панель с навигацией
st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите раздел:",
    ["📊 Загрузка данных", "🔍 Анализ данных", "🛠️ Предобработка", 
     "📈 Визуализация", "🤖 Моделирование", "📊 Оценка моделей", "🔮 Прогнозирование"]
)

# Функция для загрузки данных
@st.cache_data
def load_data():
    df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
    df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')
    return df_train, df_test

# Функция предобработки
def preprocess_features(df):
    df_processed = df.copy()
    
    # Обработка mileage
    df_processed['mileage'] = df_processed['mileage'].astype(str).str.replace('kmpl', '', regex=False)
    df_processed['mileage'] = df_processed['mileage'].str.replace('km/kg', '', regex=False)
    df_processed['mileage'] = pd.to_numeric(df_processed['mileage'], errors='coerce')
    
    # Обработка engine
    df_processed['engine'] = df_processed['engine'].astype(str).str.replace('CC', '', regex=False)
    df_processed['engine'] = pd.to_numeric(df_processed['engine'], errors='coerce')
    
    # Обработка max_power
    df_processed['max_power'] = df_processed['max_power'].astype(str).str.replace('bhp', '', regex=False)
    df_processed['max_power'] = pd.to_numeric(df_processed['max_power'], errors='coerce')
    
    # Обработка torque
    torque_series = df_processed['torque'].astype(str)
    df_processed['torque_nm'] = np.nan
    df_processed['max_torque_rpm'] = np.nan
    
    for i, value in enumerate(torque_series):
        if pd.isna(value) or value == 'nan':
            continue
            
        value = value.lower().replace(' ', '')
        numbers = re.findall(r'\d+\.?\d*', value)
        
        if len(numbers) >= 1:
            torque_val = float(numbers[0])
            if 'kgm' in value or 'kg' in value:
                torque_val = torque_val * 9.80665
            df_processed.loc[i, 'torque_nm'] = torque_val
            
        if len(numbers) >= 2:
            df_processed.loc[i, 'max_torque_rpm'] = float(numbers[1])
    
    df_processed = df_processed.drop('torque', axis=1)
    
    return df_processed

# ==================== СТРАНИЦА: ЗАГРУЗКА ДАННЫХ ====================
if page == "📊 Загрузка данных":
    st.header("📊 Загрузка и обзор данных")
    
    # Варианты загрузки данных
    data_source = st.radio(
        "Выберите источник данных:",
        ["📥 Загрузить собственный CSV файл", "📊 Использовать демо-данные"]
    )
    
    if data_source == "📥 Загрузить собственный CSV файл":
        uploaded_file = st.file_uploader("Загрузите CSV файл с данными", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df_uploaded = pd.read_csv(uploaded_file)
                st.success(f"✅ Файл успешно загружен: {len(df_uploaded)} строк, {df_uploaded.shape[1]} столбцов")
                
                # Разделение на train/test если есть целевая переменная
                if 'selling_price' in df_uploaded.columns:
                    # Для демо разделим случайно
                    np.random.seed(42)
                    mask = np.random.rand(len(df_uploaded)) < 0.8
                    df_train = df_uploaded[mask].copy()
                    df_test = df_uploaded[~mask].copy()
                    st.session_state.df_train = df_train
                    st.session_state.df_test = df_test
                    
                    st.info(f"Данные разделены: train={len(df_train)} строк, test={len(df_test)} строк")
                else:
                    st.session_state.df_train = df_uploaded
                    st.session_state.df_test = pd.DataFrame()
                    st.warning("⚠️ В данных нет целевой переменной 'selling_price'")
                
            except Exception as e:
                st.error(f"Ошибка при чтении файла: {e}")
    
    else:  # Использовать демо-данные
        if st.button("📥 Загрузить демо-данные", type="primary"):
            with st.spinner("Загрузка демо-данных..."):
                df_train, df_test = load_data()
                st.session_state.df_train = df_train
                st.session_state.df_test = df_test
                
            st.success("✅ Демо-данные успешно загружены!")
    
    # Показать данные если они загружены
    if st.session_state.df_train is not None:
        df_train = st.session_state.df_train
        df_test = st.session_state.df_test
        
        # Информация о данных
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Обучающая выборка", f"{df_train.shape[0]} строк, {df_train.shape[1]} столбцов")
        with col2:
            if len(df_test) > 0:
                st.metric("Тестовая выборка", f"{df_test.shape[0]} строк, {df_test.shape[1]} столбцов")
            else:
                st.metric("Тестовая выборка", "Не загружена")
        
        # Просмотр данных
        tab1, tab2 = st.tabs(["📋 Просмотр данных", "📊 Описательная статистика"])
        
        with tab1:
            st.subheader("Первые 10 строк данных")
            st.dataframe(df_train.head(10), use_container_width=True)
            
            if len(df_test) > 0:
                st.subheader("Тестовая выборка (первые 5 строк)")
                st.dataframe(df_test.head(5), use_container_width=True)
        
        with tab2:
            st.subheader("Описательная статистика")
            st.dataframe(df_train.describe(), use_container_width=True)
            
            st.subheader("Информация о колонках")
            col_info = pd.DataFrame({
                'Колонка': df_train.columns,
                'Тип': df_train.dtypes.astype(str),
                'Уникальных значений': df_train.nunique(),
                'Пропусков': df_train.isnull().sum()
            })
            st.dataframe(col_info, use_container_width=True)

# ==================== СТРАНИЦА: АНАЛИЗ ДАННЫХ ====================
elif page == "🔍 Анализ данных":
    st.header("🔍 Анализ качества данных")
    
    if st.session_state.df_train is None:
        st.warning("⚠️ Сначала загрузите данные на странице 'Загрузка данных'")
        st.stop()
    
    df_train = st.session_state.df_train.copy()
    
    # Анализ пропусков
    st.subheader("📊 Анализ пропущенных значений")
    
    missing_train = df_train.isnull().sum()
    missing_train = missing_train[missing_train > 0]
    
    if len(missing_train) > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Количество пропусков по колонкам:**")
            missing_df = missing_train.reset_index()
            missing_df.columns = ['Колонка', 'Пропусков']
            missing_df['Процент'] = (missing_df['Пропусков'] / len(df_train) * 100).round(2)
            st.dataframe(missing_df, use_container_width=True)
        
        with col2:
            # Визуализация пропусков
            fig, ax = plt.subplots(figsize=(6, 4))
            missing_train.plot(kind='bar', ax=ax, color='coral')
            ax.set_title('Распределение пропусков')
            ax.set_ylabel('Количество пропусков')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.success("✅ Пропусков нет")
    
    # Анализ дубликатов
    st.subheader("🔍 Поиск дубликатов")
    
    duplicates_count = df_train.duplicated().sum()
    
    if duplicates_count > 0:
        st.warning(f"⚠️ Найдено {duplicates_count} дубликатов ({duplicates_count/len(df_train)*100:.2f}%)")
        
        if st.checkbox("Показать дубликаты"):
            duplicates = df_train[df_train.duplicated(keep=False)]
            st.dataframe(duplicates.sort_values(df_train.columns.tolist()), use_container_width=True)
    else:
        st.success("✅ Дубликатов нет")
    
    # Статистика по типам данных
    st.subheader("📈 Распределение типов данных")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    dtype_counts = df_train.dtypes.value_counts()
    colors = plt.cm.Set3(range(len(dtype_counts)))
    ax.pie(dtype_counts.values, labels=dtype_counts.index.astype(str), 
           autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Распределение типов данных')
    st.pyplot(fig)
    
    # Уникальные значения в категориальных признаках
    st.subheader("🎯 Уникальные значения в категориальных признаках")
    
    categorical_cols = df_train.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            unique_vals = df_train[col].nunique()
            st.write(f"**{col}**: {unique_vals} уникальных значений")
            
            if unique_vals <= 20:  # Показываем топ значений если их немного
                value_counts = df_train[col].value_counts().head(10)
                
                fig, ax = plt.subplots(figsize=(10, 3))
                value_counts.plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title(f'Топ значений: {col}')
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

# ==================== СТРАНИЦА: ПРЕДОБРАБОТКА ====================
elif page == "🛠️ Предобработка":
    st.header("🛠️ Предобработка данных")
    
    if st.session_state.df_train is None:
        st.warning("⚠️ Сначала загрузите данные")
        st.stop()
    
    df_train = st.session_state.df_train.copy()
    df_test = st.session_state.df_test.copy() if st.session_state.df_test is not None else None
    
    st.info("Этапы предобработки данных:")
    
    # Кнопка для запуска предобработки
    if st.button("🚀 Запустить предобработку", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Шаг 1: Обработка признаков
        status_text.text("Шаг 1: Обработка признаков...")
        df_train_processed = preprocess_features(df_train)
        if df_test is not None and len(df_test) > 0:
            df_test_processed = preprocess_features(df_test)
        else:
            df_test_processed = pd.DataFrame()
        progress_bar.progress(25)
        
        # Шаг 2: Заполнение пропусков
        status_text.text("Шаг 2: Заполнение пропусков...")
        numeric_cols = ['mileage', 'engine', 'max_power', 'torque_nm', 'max_torque_rpm', 'seats']
        
        # Добавляем year и km_driven если они есть
        for col in ['year', 'km_driven', 'selling_price']:
            if col in df_train_processed.columns:
                numeric_cols.append(col)
        
        numeric_cols = [col for col in numeric_cols if col in df_train_processed.columns]
        
        train_medians = df_train_processed[numeric_cols].median()
        df_train_processed[numeric_cols] = df_train_processed[numeric_cols].fillna(train_medians)
        
        if len(df_test_processed) > 0:
            df_test_processed[numeric_cols] = df_test_processed[numeric_cols].fillna(train_medians)
        progress_bar.progress(50)
        
        # Шаг 3: Преобразование типов
        status_text.text("Шаг 3: Преобразование типов данных...")
        if 'engine' in df_train_processed.columns:
            df_train_processed['engine'] = df_train_processed['engine'].astype('int64')
        if 'seats' in df_train_processed.columns:
            df_train_processed['seats'] = df_train_processed['seats'].astype('int64')
        
        if len(df_test_processed) > 0:
            if 'engine' in df_test_processed.columns:
                df_test_processed['engine'] = df_test_processed['engine'].astype('int64')
            if 'seats' in df_test_processed.columns:
                df_test_processed['seats'] = df_test_processed['seats'].astype('int64')
        progress_bar.progress(75)
        
        # Шаг 4: Обработка брендов
        status_text.text("Шаг 4: Обработка категориальных признаков...")
        if 'name' in df_train_processed.columns:
            df_train_processed['brand'] = df_train_processed['name'].str.split().str[0]
            if len(df_test_processed) > 0:
                df_test_processed['brand'] = df_test_processed['name'].str.split().str[0]
            
            brand_counts = df_train_processed['brand'].value_counts()
            rare_brands = brand_counts[brand_counts < 10].index
            df_train_processed['brand'] = df_train_processed['brand'].replace(rare_brands, 'Other')
            
            if len(df_test_processed) > 0:
                df_test_processed['brand'] = df_test_processed['brand'].replace(rare_brands, 'Other')
        
        progress_bar.progress(100)
        status_text.text("✅ Предобработка завершена!")
        
        # Сохранение обработанных данных
        st.session_state.df_train_processed = df_train_processed
        st.session_state.df_test_processed = df_test_processed
        
        # Показать результаты
        st.subheader("📊 Результаты предобработки")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Обработанные данные (первые 5 строк):**")
            st.dataframe(df_train_processed.head(), use_container_width=True)
        
        with col2:
            st.write("**Информация о колонках:**")
            
            # Пропуски после обработки
            missing_after = df_train_processed.isnull().sum().sum()
            st.metric("Осталось пропусков", missing_after)
            
            # Типы данных
            dtypes_info = df_train_processed.dtypes.value_counts().reset_index()
            dtypes_info.columns = ['Тип данных', 'Количество колонок']
            st.dataframe(dtypes_info, use_container_width=True)
        
        st.success("✅ Данные готовы для анализа и моделирования!")

# ==================== СТРАНИЦА: ВИЗУАЛИЗАЦИЯ (EDA) ====================
elif page == "📈 Визуализация":
    st.header("📈 Визуализация данных (EDA)")
    
    if 'df_train_processed' not in st.session_state and st.session_state.df_train is not None:
        # Используем исходные данные если обработанных нет
        df_train = st.session_state.df_train.copy()
    elif 'df_train_processed' in st.session_state:
        df_train = st.session_state.df_train_processed.copy()
    else:
        st.warning("⚠️ Сначала загрузите и обработайте данные")
        st.stop()
    
    # Выбор типа визуализации
    viz_type = st.selectbox(
        "Выберите тип визуализации:",
        ["📊 Распределение целевой переменной", 
         "📈 Корреляционная матрица", 
         "🔗 Зависимости признаков", 
         "📉 Распределение признаков",
         "🎯 Категориальные признаки"]
    )
    
    if viz_type == "📊 Распределение целевой переменной":
        st.subheader("📊 Распределение цен на автомобили")
        
        if 'selling_price' not in df_train.columns:
            st.warning("⚠️ В данных нет целевой переменной 'selling_price'")
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            
            # Гистограмма
            ax1.hist(df_train['selling_price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title('Распределение цен')
            ax1.set_xlabel('Цена')
            ax1.set_ylabel('Количество')
            ax1.grid(True, alpha=0.3)
            
            # Логарифмированная гистограмма
            ax2.hist(np.log1p(df_train['selling_price']), bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
            ax2.set_title('Распределение логарифма цен')
            ax2.set_xlabel('log(Цена + 1)')
            ax2.set_ylabel('Количество')
            ax2.grid(True, alpha=0.3)
            
            # Boxplot
            ax3.boxplot(df_train['selling_price'], vert=False)
            ax3.set_title('Boxplot цен')
            ax3.set_xlabel('Цена')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Статистики
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Средняя цена", f"{df_train['selling_price'].mean():,.0f}")
            with col2:
                st.metric("Медианная цена", f"{df_train['selling_price'].median():,.0f}")
            with col3:
                st.metric("Минимальная цена", f"{df_train['selling_price'].min():,.0f}")
            with col4:
                st.metric("Максимальная цена", f"{df_train['selling_price'].max():,.0f}")
    
    elif viz_type == "📈 Корреляционная матрица":
        st.subheader("🔗 Корреляционная матрица")
        
        # Вычисляем корреляции
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df_train[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5, ax=ax, 
                       cbar_kws={"shrink": 0.8})
            ax.set_title('Корреляционная матрица числовых признаков')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Показать топ корреляций если есть цена
            if 'selling_price' in numeric_cols:
                st.subheader("📊 Топ корреляций с ценой")
                price_corr = corr_matrix['selling_price'].sort_values(ascending=False)
                price_corr_df = price_corr.reset_index()
                price_corr_df.columns = ['Признак', 'Корреляция с ценой']
                st.dataframe(price_corr_df, use_container_width=True)
        else:
            st.warning("Недостаточно числовых признаков для построения корреляционной матрицы")
    
    elif viz_type == "🔗 Зависимости признаков":
        st.subheader("📈 Зависимость цены от других признаков")
        
        if 'selling_price' not in df_train.columns:
            st.warning("⚠️ В данных нет целевой переменной 'selling_price'")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox(
                    "Выберите признак для оси X:",
                    df_train.select_dtypes(include=[np.number]).columns.tolist()
                )
            
            with col2:
                plot_type = st.selectbox(
                    "Тип графика:",
                    ["Точечный", "Линейный с доверительным интервалом"]
                )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if plot_type == "Точечный":
                sns.scatterplot(data=df_train, x=x_axis, y='selling_price', 
                               alpha=0.6, s=50, ax=ax)
            else:
                sns.regplot(data=df_train, x=x_axis, y='selling_price', 
                           scatter_kws={'alpha': 0.6, 's': 20},
                           line_kws={'color': 'red'}, ax=ax)
            
            ax.set_title(f'Зависимость цены от {x_axis}')
            ax.set_xlabel(x_axis)
            ax.set_ylabel('Цена')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    elif viz_type == "📉 Распределение признаков":
        st.subheader("📊 Распределение числовых признаков")
        
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            # Выбор признаков для визуализации
            selected_features = st.multiselect(
                "Выберите признаки для визуализации:",
                numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )
            
            if selected_features:
                n_cols = min(2, len(selected_features))
                n_rows = (len(selected_features) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
                
                for idx, feature in enumerate(selected_features):
                    if idx < len(axes):
                        axes[idx].hist(df_train[feature].dropna(), bins=30, alpha=0.7, 
                                      color='lightgreen', edgecolor='black')
                        axes[idx].set_title(f'Распределение {feature}')
                        axes[idx].set_xlabel(feature)
                        axes[idx].set_ylabel('Частота')
                        axes[idx].grid(True, alpha=0.3)
                
                # Скрыть пустые subplots
                for idx in range(len(selected_features), len(axes)):
                    axes[idx].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("Нет числовых признаков для визуализации")
    
    elif viz_type == "🎯 Категориальные признаки":
        st.subheader("📊 Анализ категориальных признаков")
        
        categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) > 0:
            selected_cat = st.selectbox(
                "Выберите категориальный признак:",
                categorical_cols
            )
            
            if selected_cat:
                # Bar plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Количество по категориям
                value_counts = df_train[selected_cat].value_counts().head(15)
                bars = ax1.bar(range(len(value_counts)), value_counts.values, color='lightblue')
                ax1.set_title(f'Распределение по {selected_cat}')
                ax1.set_xlabel(selected_cat)
                ax1.set_ylabel('Количество')
                ax1.set_xticks(range(len(value_counts)))
                ax1.set_xticklabels(value_counts.index, rotation=45, ha='right')
                
                # Добавляем значения на бары
                for bar, value in zip(bars, value_counts.values):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{value:,}', ha='center', va='bottom')
                
                # Boxplot если есть цена
                if 'selling_price' in df_train.columns:
                    # Берем топ категорий для boxplot
                    top_categories = value_counts.index.tolist()[:10]
                    boxplot_data = []
                    for cat in top_categories:
                        boxplot_data.append(df_train[df_train[selected_cat] == cat]['selling_price'].values)
                    
                    ax2.boxplot(boxplot_data, labels=top_categories)
                    ax2.set_title(f'Цены по категориям {selected_cat}')
                    ax2.set_ylabel('Цена')
                    ax2.tick_params(axis='x', rotation=45)
                    ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("Нет категориальных признаков для визуализации")

# ==================== СТРАНИЦА: МОДЕЛИРОВАНИЕ ====================
elif page == "🤖 Моделирование":
    st.header("🤖 Построение моделей машинного обучения")
    
    if 'df_train_processed' not in st.session_state and st.session_state.df_train is not None:
        # Если предобработка не выполнена, используем исходные данные
        df_train = st.session_state.df_train.copy()
        df_test = st.session_state.df_test.copy() if st.session_state.df_test is not None else pd.DataFrame()
    elif 'df_train_processed' in st.session_state:
        df_train = st.session_state.df_train_processed
        df_test = st.session_state.df_test_processed if hasattr(st.session_state, 'df_test_processed') else pd.DataFrame()
    else:
        st.warning("⚠️ Сначала загрузите и обработайте данные")
        st.stop()
    
    if 'selling_price' not in df_train.columns:
        st.error("❌ В данных отсутствует целевая переменная 'selling_price'")
        st.stop()
    
    # Подготовка данных
    st.subheader("📊 Подготовка данных для моделирования")
    
    # Выбор признаков
    numeric_columns = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    features = [col for col in numeric_columns if col != 'selling_price']
    
    X_train = df_train[features].copy()
    y_train = df_train['selling_price'].copy()
    
    if len(df_test) > 0 and 'selling_price' in df_test.columns:
        X_test = df_test[features].copy()
        y_test = df_test['selling_price'].copy()
    else:
        # Если тестовых данных нет, разделяем обучающие
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
    
    # Сохраняем признаки для использования при прогнозировании
    st.session_state.features = features
    
    # Стандартизация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Сохраняем scaler для использования при прогнозировании
    st.session_state.scaler = scaler
    
    st.success(f"✅ Данные подготовлены: {X_train.shape[1]} признаков, {X_train.shape[0]} образцов")
    
    # Информация о данных
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Обучающая выборка", f"{X_train.shape[0]} образцов")
    with col2:
        st.metric("Тестовая выборка", f"{X_test.shape[0]} образцов")
    with col3:
        st.metric("Количество признаков", f"{X_train.shape[1]}")
    
    # Выбор моделей для обучения
    st.subheader("🎯 Выбор и обучение моделей")
    
    selected_models = st.multiselect(
        "Выберите модели для обучения:",
        ["Линейная регрессия", "Lasso", "Ridge", "ElasticNet"],
        default=["Линейная регрессия", "Lasso"]
    )
    
    # Настройки моделей
    st.subheader("⚙️ Настройки моделей")
    
    model_params = {}
    
    if "Lasso" in selected_models:
        alpha_lasso = st.slider("Alpha для Lasso", 0.01, 10.0, 1.0, 0.01)
        model_params['Lasso'] = {'alpha': alpha_lasso}
    
    if "Ridge" in selected_models:
        alpha_ridge = st.slider("Alpha для Ridge", 0.01, 10.0, 1.0, 0.01)
        model_params['Ridge'] = {'alpha': alpha_ridge}
    
    if "ElasticNet" in selected_models:
        col1, col2 = st.columns(2)
        with col1:
            alpha_elastic = st.slider("Alpha для ElasticNet", 0.01, 10.0, 0.1, 0.01)
        with col2:
            l1_ratio = st.slider("L1 ratio для ElasticNet", 0.0, 1.0, 0.5, 0.01)
        model_params['ElasticNet'] = {'alpha': alpha_elastic, 'l1_ratio': l1_ratio}
    
    # Отдельная кнопка для обучения моделей
    if st.button("🚀 Обучить выбранные модели", type="primary", key="train_models"):
        with st.spinner("Обучение моделей..."):
            results = []
            
            # Линейная регрессия
            if "Линейная регрессия" in selected_models:
                lr_model = LinearRegression()
                lr_model.fit(X_train_scaled_df, y_train)
                y_pred_train_lr = lr_model.predict(X_train_scaled_df)
                y_pred_test_lr = lr_model.predict(X_test_scaled_df)
                
                results.append({
                    "Модель": "Линейная регрессия",
                    "R² (train)": r2_score(y_train, y_pred_train_lr),
                    "R² (test)": r2_score(y_test, y_pred_test_lr),
                    "RMSE (test)": np.sqrt(mean_squared_error(y_test, y_pred_test_lr)),
                    "MAE (test)": mean_absolute_error(y_test, y_pred_test_lr)
                })
                
                st.session_state.models["Линейная регрессия"] = {
                    "model": lr_model,
                    "predictions": y_pred_test_lr,
                    "coefs": lr_model.coef_,
                    "intercept": lr_model.intercept_
                }
            
            # Lasso
            if "Lasso" in selected_models:
                lasso_params = model_params.get('Lasso', {'alpha': 1.0})
                lasso_model = Lasso(**lasso_params)
                lasso_model.fit(X_train_scaled_df, y_train)
                y_pred_train_lasso = lasso_model.predict(X_train_scaled_df)
                y_pred_test_lasso = lasso_model.predict(X_test_scaled_df)
                
                results.append({
                    "Модель": "Lasso",
                    "R² (train)": r2_score(y_train, y_pred_train_lasso),
                    "R² (test)": r2_score(y_test, y_pred_test_lasso),
                    "RMSE (test)": np.sqrt(mean_squared_error(y_test, y_pred_test_lasso)),
                    "MAE (test)": mean_absolute_error(y_test, y_pred_test_lasso)
                })
                
                st.session_state.models["Lasso"] = {
                    "model": lasso_model,
                    "predictions": y_pred_test_lasso,
                    "coefs": lasso_model.coef_,
                    "intercept": lasso_model.intercept_
                }
            
            # Ridge
            if "Ridge" in selected_models:
                ridge_params = model_params.get('Ridge', {'alpha': 1.0})
                ridge_model = Ridge(**ridge_params)
                ridge_model.fit(X_train_scaled_df, y_train)
                y_pred_train_ridge = ridge_model.predict(X_train_scaled_df)
                y_pred_test_ridge = ridge_model.predict(X_test_scaled_df)
                
                results.append({
                    "Модель": "Ridge",
                    "R² (train)": r2_score(y_train, y_pred_train_ridge),
                    "R² (test)": r2_score(y_test, y_pred_test_ridge),
                    "RMSE (test)": np.sqrt(mean_squared_error(y_test, y_pred_test_ridge)),
                    "MAE (test)": mean_absolute_error(y_test, y_pred_test_ridge)
                })
                
                st.session_state.models["Ridge"] = {
                    "model": ridge_model,
                    "predictions": y_pred_test_ridge,
                    "coefs": ridge_model.coef_,
                    "intercept": ridge_model.intercept_
                }
            
            # ElasticNet
            if "ElasticNet" in selected_models:
                elastic_params = model_params.get('ElasticNet', {'alpha': 0.1, 'l1_ratio': 0.5})
                elastic_model = ElasticNet(**elastic_params)
                elastic_model.fit(X_train_scaled_df, y_train)
                y_pred_train_elastic = elastic_model.predict(X_train_scaled_df)
                y_pred_test_elastic = elastic_model.predict(X_test_scaled_df)
                
                results.append({
                    "Модель": "ElasticNet",
                    "R² (train)": r2_score(y_train, y_pred_train_elastic),
                    "R² (test)": r2_score(y_test, y_pred_test_elastic),
                    "RMSE (test)": np.sqrt(mean_squared_error(y_test, y_pred_test_elastic)),
                    "MAE (test)": mean_absolute_error(y_test, y_pred_test_elastic)
                })
                
                st.session_state.models["ElasticNet"] = {
                    "model": elastic_model,
                    "predictions": y_pred_test_elastic,
                    "coefs": elastic_model.coef_,
                    "intercept": elastic_model.intercept_
                }
            
            # Устанавливаем флаг, что модель обучена
            st.session_state.model_trained = True
            
            # Выбираем лучшую модель по R²
            if results:
                best_model_info = max(results, key=lambda x: x['R² (test)'])
                best_model_name = best_model_info['Модель']
                st.session_state.current_model = best_model_name
                
                st.success(f"✅ Модели успешно обучены! Лучшая модель: {best_model_name} (R²={best_model_info['R² (test)']:.4f})")
            
            # Показать результаты
            st.subheader("📊 Результаты обучения моделей")
            results_df = pd.DataFrame(results)
            
            # Форматирование числовых значений
            display_df = results_df.copy()
            for col in ['R² (train)', 'R² (test)']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            for col in ['RMSE (test)', 'MAE (test)']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}")
            
            st.dataframe(display_df.style.highlight_max(subset=['R² (test)'], color='lightgreen')
                                  .highlight_min(subset=['RMSE (test)', 'MAE (test)'], color='lightcoral'),
                        use_container_width=True)
    
    # ОТДЕЛЬНАЯ СЕКЦИЯ ДЛЯ СОХРАНЕНИЯ МОДЕЛИ (после обучения)
    if st.session_state.model_trained and st.session_state.models:
        st.subheader("💾 Сохранение модели")
        
        # Выбор модели для сохранения
        model_to_save = st.selectbox(
            "Выберите модель для сохранения:",
            list(st.session_state.models.keys()),
            key="save_model_select"
        )
        
        # Имя файла для сохранения
        filename = st.text_input("Имя файла для сохранения:", value=f"{model_to_save.lower()}_model.pkl")
        
        # Кнопка сохранения
        if st.button("💾 Сохранить выбранную модель", type="primary"):
            try:
                if model_to_save in st.session_state.models:
                    model_info = st.session_state.models[model_to_save]
                    
                    from sklearn.pipeline import Pipeline
                    
                    # Создаем пайплайн
                    pipeline = Pipeline(steps=[
                        ('scaler', st.session_state.scaler),
                        ('model', model_info['model'])
                    ])
                    
                    # Сохраняем пайплайн и признаки
                    joblib.dump(pipeline, filename)
                    joblib.dump(st.session_state.features, f"{filename.split('.')[0]}_features.pkl")
                    
                    st.success(f"✅ Модель '{model_to_save}' сохранена в файл '{filename}'")
                    st.success(f"✅ Признаки сохранены в файл '{filename.split('.')[0]}_features.pkl'")
                    
                    # Информация о сохраненной модели
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Файл модели:** {filename}")
                        st.info(f"**Тип модели:** {type(model_info['model']).__name__}")
                    with col2:
                        st.info(f"**Файл признаков:** {filename.split('.')[0]}_features.pkl")
                        st.info(f"**Количество признаков:** {len(st.session_state.features)}")
                    
                    # Показать список сохраненных файлов
                    import os
                    if os.path.exists(filename):
                        st.write("**Сохраненные файлы в текущей директории:**")
                        files = [f for f in os.listdir('.') if f.endswith('.pkl')]
                        for file in files:
                            size = os.path.getsize(file)
                            st.write(f"- {file} ({size:,} байт)")
                            
            except Exception as e:
                st.error(f"❌ Ошибка при сохранении модели: {e}")
    
    elif not st.session_state.model_trained:
        st.info("ℹ️ Обучите модели, чтобы появилась возможность их сохранить")

# ==================== СТРАНИЦА: ОЦЕНКА МОДЕЛЕЙ ====================
elif page == "📊 Оценка моделей":
    st.header("📊 Сравнение и оценка моделей")
    
    if not st.session_state.models:
        st.warning("⚠️ Сначала обучите модели на странице 'Моделирование'")
        st.stop()
    
    if 'df_train_processed' in st.session_state:
        df_test = st.session_state.df_test_processed
    elif st.session_state.df_test is not None:
        df_test = st.session_state.df_test
    else:
        st.warning("⚠️ Нет тестовых данных для оценки")
        st.stop()
    
    if 'selling_price' not in df_test.columns:
        st.warning("⚠️ В тестовых данных нет целевой переменной 'selling_price'")
        st.stop()
    
    y_test = df_test['selling_price'].copy()
    
    # Сравнение моделей
    st.subheader("📈 Сравнение производительности моделей")
    
    # Метрики
    metrics_data = []
    for model_name, model_info in st.session_state.models.items():
        y_pred = model_info["predictions"]
        
        metrics_data.append({
            "Модель": model_name,
            "R²": r2_score(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred),
            "MAPE (%)": np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Форматирование для отображения
    display_metrics = metrics_df.copy()
    display_metrics['R²'] = display_metrics['R²'].apply(lambda x: f"{x:.4f}")
    display_metrics['RMSE'] = display_metrics['RMSE'].apply(lambda x: f"{x:,.0f}")
    display_metrics['MAE'] = display_metrics['MAE'].apply(lambda x: f"{x:,.0f}")
    display_metrics['MAPE (%)'] = display_metrics['MAPE (%)'].apply(lambda x: f"{x:.2f}")
    
    # Отображение метрик
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(display_metrics.style.highlight_max(subset=['R²'], color='lightgreen')
                               .highlight_min(subset=['RMSE', 'MAE', 'MAPE (%)'], color='lightcoral'),
                    use_container_width=True)
    
    with col2:
        best_model = metrics_df.loc[metrics_df['R²'].idxmax(), 'Модель']
        best_r2 = metrics_df.loc[metrics_df['R²'].idxmax(), 'R²']
        best_rmse = metrics_df.loc[metrics_df['RMSE'].idxmin(), 'RMSE']
        
        st.metric("Лучшая модель", best_model)
        st.metric("Лучший R²", f"{best_r2:.4f}")
        st.metric("Лучший RMSE", f"{best_rmse:,.0f}")
    
    # Визуализация прогнозов
    st.subheader("📊 Визуализация прогнозов")
    
    model_to_plot = st.selectbox(
        "Выберите модель для визуализации:",
        list(st.session_state.models.keys())
    )
    
    if model_to_plot:
        model_info = st.session_state.models[model_to_plot]
        y_pred = model_info["predictions"]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # График фактических vs прогнозных значений
        ax1.scatter(y_test, y_pred, alpha=0.5)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Фактические значения')
        ax1.set_ylabel('Прогнозные значения')
        ax1.set_title(f'{model_to_plot}: Фактические vs Прогнозные значения')
        ax1.grid(True, alpha=0.3)
        
        # График остатков
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Прогнозные значения')
        ax2.set_ylabel('Остатки')
        ax2.set_title(f'{model_to_plot}: График остатков')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # ВИЗУАЛИЗАЦИЯ ВЕСОВ МОДЕЛИ (ТРЕБОВАНИЕ ЗАДАНИЯ)
    st.subheader("🔍 Визуализация весов модели")
    
    model_for_weights = st.selectbox(
        "Выберите модель для анализа весов:",
        list(st.session_state.models.keys()),
        key="weights_model"
    )
    
    if model_for_weights:
        model_info = st.session_state.models[model_for_weights]
        coefs = model_info["coefs"]
        
        if st.session_state.features and len(coefs) == len(st.session_state.features):
            # Создаем DataFrame с весами
            weights_df = pd.DataFrame({
                'Признак': st.session_state.features,
                'Вес': coefs,
                'Абсолютное значение': np.abs(coefs)
            }).sort_values('Абсолютное значение', ascending=False)
            
            # Ограничиваем количество признаков для лучшей визуализации
            top_n = st.slider("Количество признаков для отображения:", 5, 20, 10)
            weights_display = weights_df.head(top_n)
            
            # График весов признаков
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Bar plot весов
            colors = ['red' if x < 0 else 'blue' for x in weights_display['Вес']]
            y_pos = np.arange(len(weights_display))
            ax1.barh(y_pos, weights_display['Вес'], color=colors)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(weights_display['Признак'])
            ax1.set_xlabel('Значение веса')
            ax1.set_title(f'Веса признаков ({model_for_weights})')
            ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Pie chart значимости
            top_weights = weights_display.head(6)
            ax2.pie(top_weights['Абсолютное значение'], 
                   labels=top_weights['Признак'],
                   autopct='%1.1f%%',
                   startangle=90,
                   colors=plt.cm.Set3(np.arange(len(top_weights))))
            ax2.set_title('Относительная важность признаков (топ-6)')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Таблица с весами
            st.write("**Полная таблица весов:**")
            st.dataframe(weights_df, use_container_width=True)
            
            # Статистика по весам
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Всего признаков", len(weights_df))
            with col2:
                st.metric("Положительных весов", (weights_df['Вес'] > 0).sum())
            with col3:
                st.metric("Отрицательных весов", (weights_df['Вес'] < 0).sum())
            with col4:
                st.metric("Нулевых весов", (weights_df['Вес'] == 0).sum())

# ==================== СТРАНИЦА: ПРОГНОЗИРОВАНИЕ ====================
elif page == "🔮 Прогнозирование":
    st.header("🔮 Прогнозирование цен на автомобили")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Загрузка сохраненной модели")

    # Добавляем выбор файла для загрузки
    model_files = [f for f in os.listdir('.') if f.endswith('.pkl') and 'features' not in f]

    if model_files:
        selected_model_file = st.sidebar.selectbox(
            "Выберите файл модели:",
            model_files,
            key="model_file_select"
        )
        
        if st.sidebar.button("📥 Загрузить сохраненную модель"):
            try:
                # Определяем имя файла с признаками
                features_file = selected_model_file.replace('.pkl', '_features.pkl')
                
                if os.path.exists(selected_model_file) and os.path.exists(features_file):
                    pipeline = joblib.load(selected_model_file)
                    features = joblib.load(features_file)
                    
                    # Сохраняем в session_state
                    st.session_state.saved_pipeline = pipeline
                    st.session_state.saved_features = features
                    st.sidebar.success(f"✅ Модель '{selected_model_file}' загружена!")
                    
                    # Показываем информацию о модели
                    st.sidebar.write(f"**Модель:** {type(pipeline.named_steps['model']).__name__}")
                    st.sidebar.write(f"**Признаков:** {len(features)}")
                    st.sidebar.write(f"**Файл признаков:** {features_file}")
                else:
                    st.sidebar.error(f"❌ Файл признаков '{features_file}' не найден")
                    
            except Exception as e:
                st.sidebar.error(f"❌ Ошибка при загрузке: {e}")
    else:
        st.sidebar.info("ℹ️ Нет сохраненных моделей. Сначала обучите и сохраните модель.")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Сначала обучите модель на странице 'Моделирование'")
        st.stop()
    
    if not st.session_state.models:
        st.warning("⚠️ Нет обученных моделей")
        st.stop()
    
    # Выбор способа ввода данных
    input_method = st.radio(
        "Выберите способ ввода данных:",
        ["📝 Ручной ввод признаков", "📁 Загрузка CSV файла"]
    )
    
    if input_method == "📝 Ручной ввод признаков":
        st.subheader("📝 Введите значения признаков")
        
        # Создаем форму для ввода признаков
        if st.session_state.features:
            # Группируем признаки для удобства
            feature_values = {}
            
            # Разделяем признаки на группы
            basic_features = ['year', 'km_driven', 'mileage']
            engine_features = ['engine', 'max_power', 'torque_nm', 'max_torque_rpm']
            other_features = [f for f in st.session_state.features if f not in basic_features + engine_features]
            
            # Базовые признаки
            st.write("**Основные параметры:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                if 'year' in st.session_state.features:
                    feature_values['year'] = st.number_input("Год выпуска", min_value=1990, max_value=2024, value=2015)
            with col2:
                if 'km_driven' in st.session_state.features:
                    feature_values['km_driven'] = st.number_input("Пробег (км)", min_value=0, max_value=500000, value=50000, step=1000)
            with col3:
                if 'mileage' in st.session_state.features:
                    feature_values['mileage'] = st.number_input("Расход топлива", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
            
            # Параметры двигателя
            st.write("**Параметры двигателя:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if 'engine' in st.session_state.features:
                    feature_values['engine'] = st.number_input("Объем двигателя (CC)", min_value=500, max_value=5000, value=1500, step=100)
            with col2:
                if 'max_power' in st.session_state.features:
                    feature_values['max_power'] = st.number_input("Мощность (bhp)", min_value=0.0, max_value=500.0, value=100.0, step=1.0)
            with col3:
                if 'torque_nm' in st.session_state.features:
                    feature_values['torque_nm'] = st.number_input("Крутящий момент (Nm)", min_value=0.0, max_value=1000.0, value=200.0, step=10.0)
            with col4:
                if 'max_torque_rpm' in st.session_state.features:
                    feature_values['max_torque_rpm'] = st.number_input("Обороты крутящего момента", min_value=1000, max_value=10000, value=3000, step=100)
            
            # Остальные признаки
            if other_features:
                st.write("**Другие параметры:**")
                for feature in other_features:
                    if feature == 'seats':
                        feature_values['seats'] = st.selectbox("Количество мест", [2, 4, 5, 6, 7, 8, 9])
                    else:
                        # Для числовых признаков
                        if feature in st.session_state.features:
                            feature_values[feature] = st.number_input(feature, value=0.0)
            
            # Выбор модели для прогноза
            st.subheader("🎯 Выбор модели для прогноза")
            model_for_prediction = st.selectbox(
                "Выберите модель:",
                list(st.session_state.models.keys())
            )
            
            if st.button("🔮 Сделать прогноз", type="primary"):
                try:
                    # Создаем DataFrame с введенными признаками
                    input_df = pd.DataFrame([feature_values])
                    
                    # Добавляем недостающие признаки с нулевыми значениями
                    for feature in st.session_state.features:
                        if feature not in input_df.columns:
                            input_df[feature] = 0
                    
                    # Упорядочиваем признаки как при обучении
                    input_df = input_df[st.session_state.features]
                    
                    # Применяем стандартизацию
                    if st.session_state.scaler:
                        input_scaled = st.session_state.scaler.transform(input_df)
                    else:
                        input_scaled = input_df.values
                    
                    # Получаем модель
                    model_info = st.session_state.models[model_for_prediction]
                    model = model_info["model"]
                    
                    # Делаем прогноз
                    prediction = model.predict(input_scaled)[0]
                    
                    # Отображаем результат
                    st.success(f"✅ Прогноз выполнен успешно!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Прогнозируемая цена", f"{prediction:,.0f}")
                    with col2:
                        # Пример доверительного интервала
                        confidence = 0.95
                        margin = prediction * 0.1  # 10% в качестве примера
                        st.metric("Доверительный интервал", f"{prediction-margin:,.0f} - {prediction+margin:,.0f}")
                    with col3:
                        st.metric("Использована модель", model_for_prediction)
                    
                    # Визуализация вклада признаков
                    st.subheader("📊 Вклад признаков в прогноз")
                    
                    if hasattr(model, 'coef_'):
                        coefs = model.coef_
                        if hasattr(model, 'intercept_'):
                            intercept = model.intercept_
                        
                        # Вычисляем вклад каждого признака
                        contributions = coefs * input_scaled[0]
                        
                        # Создаем DataFrame для визуализации
                        contrib_df = pd.DataFrame({
                            'Признак': st.session_state.features,
                            'Значение': input_df.values[0],
                            'Вес': coefs,
                            'Вклад': contributions
                        }).sort_values('Вклад', key=abs, ascending=False)
                        
                        # Отображаем топ вкладов
                        fig, ax = plt.subplots(figsize=(10, 6))
                        top_contrib = contrib_df.head(10)
                        colors = ['green' if x > 0 else 'red' for x in top_contrib['Вклад']]
                        bars = ax.barh(range(len(top_contrib)), top_contrib['Вклад'], color=colors)
                        ax.set_yticks(range(len(top_contrib)))
                        ax.set_yticklabels(top_contrib['Признак'])
                        ax.set_xlabel('Вклад в прогноз (в единицах цены)')
                        ax.set_title('Топ-10 признаков по влиянию на прогноз')
                        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                        ax.grid(True, alpha=0.3, axis='x')
                        
                        # Добавляем значения на бары
                        for bar, value in zip(bars, top_contrib['Вклад']):
                            width = bar.get_width()
                            ax.text(width if width >= 0 else width - abs(width)*0.1, 
                                   bar.get_y() + bar.get_height()/2,
                                   f'{value:,.0f}', 
                                   ha='left' if width >= 0 else 'right', 
                                   va='center')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Таца с детальной информацией
                        st.write("**Детальная информация по вкладам:**")
                        display_contrib = contrib_df.copy()
                        display_contrib['Вклад (%)'] = (display_contrib['Вклад'] / prediction * 100).round(2)
                        st.dataframe(display_contrib, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Ошибка при прогнозировании: {e}")
    
    else:  # Загрузка CSV файла
        st.subheader("📁 Загрузка CSV файла для прогнозирования")
        
        uploaded_file = st.file_uploader("Загрузите CSV файл с данными для прогноза", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Загружаем данные
                input_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Файл загружен: {len(input_data)} строк, {input_data.shape[1]} столбцов")
                
                # Показываем первые строки
                st.write("**Первые 5 строк загруженных данных:**")
                st.dataframe(input_data.head(), use_container_width=True)
                
                # Выбор модели
                model_for_prediction = st.selectbox(
                    "Выберите модель для прогноза:",
                    list(st.session_state.models.keys()),
                    key="csv_model"
                )
                
                if st.button("🔮 Прогнозировать для всех строк", type="primary"):
                    try:
                        # Предобработка входных данных
                        if 'df_train_processed' in st.session_state:
                            # Применяем ту же предобработку, что и к обучающим данным
                            input_processed = preprocess_features(input_data)
                            
                            # Заполняем пропуски
                            numeric_cols = input_processed.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                input_processed[numeric_cols] = input_processed[numeric_cols].fillna(
                                    input_processed[numeric_cols].median()
                                )
                        else:
                            input_processed = input_data
                        
                        # Готовим признаки
                        if st.session_state.features:
                            # Добавляем недостающие признаки
                            for feature in st.session_state.features:
                                if feature not in input_processed.columns:
                                    input_processed[feature] = 0
                            
                            X_input = input_processed[st.session_state.features].copy()
                            
                            # Применяем стандартизацию
                            if st.session_state.scaler:
                                X_input_scaled = st.session_state.scaler.transform(X_input)
                            else:
                                X_input_scaled = X_input.values
                            
                            # Получаем модель и делаем прогнозы
                            model_info = st.session_state.models[model_for_prediction]
                            model = model_info["model"]
                            predictions = model.predict(X_input_scaled)
                            
                            # Добавляем прогнозы к данным
                            result_df = input_data.copy()
                            result_df['predicted_price'] = predictions
                            
                            # Сохраняем результаты
                            csv = result_df.to_csv(index=False).encode('utf-8')
                            
                            st.success(f"✅ Прогнозы успешно выполнены для {len(predictions)} строк!")
                            
                            # Показываем результаты
                            st.write("**Результаты прогнозирования (первые 10 строк):**")
                            st.dataframe(result_df.head(10), use_container_width=True)
                            
                            # Статистика прогнозов
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Средняя цена", f"{result_df['predicted_price'].mean():,.0f}")
                            with col2:
                                st.metric("Минимальная цена", f"{result_df['predicted_price'].min():,.0f}")
                            with col3:
                                st.metric("Максимальная цена", f"{result_df['predicted_price'].max():,.0f}")
                            with col4:
                                st.metric("Стандартное отклонение", f"{result_df['predicted_price'].std():,.0f}")
                            
                            # Визуализация распределения прогнозов
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.hist(result_df['predicted_price'], bins=30, alpha=0.7, color='purple', edgecolor='black')
                            ax.set_title('Распределение прогнозируемых цен')
                            ax.set_xlabel('Прогнозируемая цена')
                            ax.set_ylabel('Количество')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            
                            # Кнопка для скачивания результатов
                            st.download_button(
                                label="📥 Скачать результаты прогнозирования (CSV)",
                                data=csv,
                                file_name=f"predictions_{model_for_prediction}.csv",
                                mime="text/csv"
                            )
                        
                    except Exception as e:
                        st.error(f"Ошибка при обработке данных: {e}")
                        
            except Exception as e:
                st.error(f"Ошибка при чтении файла: {e}")

# Информация в подвале
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Информация о приложении:**
    - Цель: Прогнозирование цен на автомобили
    - Автор: Кондаков Владислав
    """
)