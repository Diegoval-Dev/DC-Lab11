import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import json
from datetime import datetime, timedelta
import numpy as np
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Análisis de Tráfico GT - Panel Ejecutivo",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-weight: 700;
    }
    .metric-card {
        background: black;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
    }
    .stMetric {
        background: black;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        color: #1f77b4;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Color palette
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'neutral': '#7f7f7f',
    'background': '#f8f9fa'
}

from data_processor import load_and_process_tweets

@st.cache_data
def load_data():
    """Load and preprocess the traffic data"""
    return load_and_process_tweets()

def create_timeline_chart(df, date_range):
    """Create interactive timeline visualization"""
    # Filter data by date range
    filtered_df = df[
        (df['created_at'].dt.date >= date_range[0]) &
        (df['created_at'].dt.date <= date_range[1])
    ]

    # Group by date
    daily_stats = filtered_df.groupby('date').agg({
        'text_clean': 'count',
        'retweet_count': 'sum',
        'like_count': 'sum',
        'username': 'nunique'
    }).reset_index()

    daily_stats.columns = ['date', 'tweet_count', 'total_retweets', 'total_likes', 'unique_users']

    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Volumen Diario de Tweets', 'Métricas de Interacción'),
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )

    # Tweet volume
    fig.add_trace(
        go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['tweet_count'],
            mode='lines+markers',
            name='Conteo de Tweets',
            line=dict(color=COLORS['primary'], width=3),
            marker=dict(size=6)
        ),
        row=1, col=1
    )

    # Engagement metrics
    fig.add_trace(
        go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['total_retweets'],
            mode='lines',
            name='Retweets',
            line=dict(color=COLORS['secondary'], width=2)
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['total_likes'],
            mode='lines',
            name='Me Gusta',
            line=dict(color=COLORS['success'], width=2),
            yaxis='y4'
        ),
        row=2, col=1, secondary_y=True
    )

    fig.update_layout(
        height=600,
        title="Línea de Tiempo de Actividad de Tweets de Tráfico",
        hovermode='x unified',
        showlegend=True
    )

    fig.update_xaxes(title_text="Fecha", row=2, col=1)
    fig.update_yaxes(title_text="Conteo de Tweets", row=1, col=1)
    fig.update_yaxes(title_text="Retweets", row=2, col=1)
    fig.update_yaxes(title_text="Me Gusta", row=2, col=1, secondary_y=True)

    return fig

def create_activity_heatmap(df):
    """Create activity heatmap by hour and day"""
    # Create hour-day activity matrix
    activity_matrix = df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')

    # Pivot for heatmap
    heatmap_data = activity_matrix.pivot(index='day_of_week', columns='hour', values='count').fillna(0)

    # Reorder days
    day_order = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    heatmap_data = heatmap_data.reindex(day_order)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Blues',
        hoverongaps=False,
        hovertemplate='<b>%{y}</b><br>Hora: %{x}<br>Tweets: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title='Mapa de Calor de Actividad de Tweets (Día vs Hora)',
        xaxis_title='Hora del Día',
        yaxis_title='Día de la Semana',
        height=400
    )

    return fig

def create_top_users_chart(df, top_n=10):
    """Create top users visualization"""
    user_stats = df.groupby('username').agg({
        'text_clean': 'count',
        'retweet_count': 'sum',
        'like_count': 'sum'
    }).reset_index()

    user_stats.columns = ['username', 'tweet_count', 'total_retweets', 'total_likes']
    user_stats = user_stats.sort_values('tweet_count', ascending=False).head(top_n)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=user_stats['username'],
        y=user_stats['tweet_count'],
        name='Conteo de Tweets',
        marker_color=COLORS['primary'],
        hovertemplate='<b>%{x}</b><br>Tweets: %{y}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Top {top_n} Usuarios Más Activos',
        xaxis_title='Usuario',
        yaxis_title='Conteo de Tweets',
        height=400,
        xaxis_tickangle=-45
    )

    return fig

def create_engagement_scatter(df):
    """Create engagement scatter plot"""
    # Calculate engagement rate if not exists
    if 'engagement_rate' not in df.columns:
        df['engagement_rate'] = (df['retweet_count'] + df['like_count'] + df['reply_count']) / (df['word_count'] + 1)

    fig = px.scatter(
        df.sample(min(1000, len(df))),  # Sample for performance
        x='word_count',
        y='engagement_rate',
        size='like_count',
        color='hour',
        hover_data=['username', 'retweet_count'],
        title='Interacción de Tweets vs Conteo de Palabras',
        labels={
            'word_count': 'Conteo de Palabras',
            'engagement_rate': 'Tasa de Interacción',
            'hour': 'Hora'
        },
        color_continuous_scale='Viridis'
    )

    fig.update_layout(height=500)

    return fig

def create_traffic_category_chart(df):
    """Create traffic category distribution chart"""
    if 'traffic_category' not in df.columns:
        return go.Figure().add_annotation(text="Categorías de tráfico no disponibles",
                                        showarrow=False, x=0.5, y=0.5)

    # Category distribution
    category_counts = df['traffic_category'].value_counts()

    # Create donut chart
    fig = go.Figure(data=[go.Pie(
        labels=category_counts.index,
        values=category_counts.values,
        hole=0.3,
        marker_colors=[COLORS['primary'], COLORS['secondary'], COLORS['success'],
                      COLORS['danger'], COLORS['neutral'], '#9467bd']
    )])

    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )

    fig.update_layout(
        title='Distribución de Categorías de Contenido de Tráfico',
        height=400,
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01)
    )

    return fig

def create_engagement_trends(df):
    """Create engagement trends over time"""
    # Group by date and calculate average engagement
    daily_engagement = df.groupby('date').agg({
        'retweet_count': 'mean',
        'like_count': 'mean',
        'reply_count': 'mean',
        'engagement_total': 'mean'
    }).reset_index()

    fig = go.Figure()

    # Add traces for different engagement types
    fig.add_trace(go.Scatter(
        x=daily_engagement['date'],
        y=daily_engagement['retweet_count'],
        mode='lines+markers',
        name='Promedio Retweets',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=4)
    ))

    fig.add_trace(go.Scatter(
        x=daily_engagement['date'],
        y=daily_engagement['like_count'],
        mode='lines+markers',
        name='Promedio Me Gusta',
        line=dict(color=COLORS['secondary'], width=2),
        marker=dict(size=4)
    ))

    fig.add_trace(go.Scatter(
        x=daily_engagement['date'],
        y=daily_engagement['reply_count'],
        mode='lines+markers',
        name='Promedio Respuestas',
        line=dict(color=COLORS['success'], width=2),
        marker=dict(size=4)
    ))

    fig.update_layout(
        title='Tendencias Diarias de Interacción Promedio',
        xaxis_title='Fecha',
        yaxis_title='Conteo Promedio',
        height=400,
        hovermode='x unified',
        showlegend=True
    )

    return fig

def create_geo_heatmap(df):
    """Create geographic heatmap (synthetic if no coords)"""
    if 'lat' not in df.columns or 'lon' not in df.columns:
        np.random.seed(42)
        df['lat'] = np.random.uniform(14.58, 14.70, len(df))
        df['lon'] = np.random.uniform(-90.60, -90.45, len(df))
    
    fig = px.density_mapbox(
        df,
        lat='lat',
        lon='lon',
        z='retweet_count',
        radius=8,
        center=dict(lat=14.63, lon=-90.53),
        zoom=11,
        mapbox_style='carto-darkmatter',
        color_continuous_scale='plasma',
        title='Mapa de Calor de Tweets de Tráfico en Ciudad de Guatemala'
    )
    fig.update_layout(height=500)
    return fig

def create_wordcloud(df):
    """Generate word cloud from tweet text"""
    text = ' '.join(df['text_clean'].dropna().astype(str))
    wc = WordCloud(
        width=800,
        height=400,
        background_color='black',
        colormap='plasma',
        max_words=150,
        collocations=False
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def train_models(df):
    """Train and evaluate 3 simple classification models on traffic_category"""
    if 'traffic_category' not in df.columns:
        st.warning("No se encontró la columna 'traffic_category'. No se pueden entrenar modelos.")
        return None, None

    # Seleccionamos características básicas
    X = df[['retweet_count', 'like_count', 'reply_count', 'word_count']]
    y = df['traffic_category']

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    models = {
        "Regresión Logística": LogisticRegression(max_iter=300),
        "Bosque Aleatorio": RandomForestClassifier(n_estimators=150, random_state=42),
        "SVM Lineal": SVC(kernel='linear', probability=True, random_state=42)
    }

    results = []
    preds = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        results.append({
            "Modelo": name,
            "Accuracy": round(acc, 3),
            "Precision": round(report['weighted avg']['precision'], 3),
            "Recall": round(report['weighted avg']['recall'], 3),
            "F1-score": round(report['weighted avg']['f1-score'], 3)
        })
        preds[name] = (y_test, y_pred)

    results_df = pd.DataFrame(results)
    return results_df, preds

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix using seaborn heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusión - {model_name}")
    st.pyplot(fig)


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Análisis de Tráfico GT - Panel Ejecutivo</h1>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("Cargando datos de tráfico..."):
        df = load_data()

    # Sidebar filters
    st.sidebar.header("Controles del Panel")

    # Date range filter
    if 'created_at' in df.columns and not df['created_at'].isna().all():
        min_date = df['created_at'].dt.date.min()
        max_date = df['created_at'].dt.date.max()

        date_range = st.sidebar.date_input(
            "Seleccionar Rango de Fechas",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if len(date_range) != 2:
            date_range = (min_date, max_date)
    else:
        date_range = (datetime.now().date() - timedelta(days=30), datetime.now().date())

    # User filter
    top_users = df['username'].value_counts().head(20).index.tolist()
    selected_users = st.sidebar.multiselect(
        "Filtrar por Usuarios",
        options=top_users,
        default=top_users[:5] if len(top_users) >= 5 else top_users
    )

    # Apply filters
    if selected_users:
        df_filtered = df[df['username'].isin(selected_users)]
    else:
        df_filtered = df

    # Overview metrics
    st.markdown('<h2 class="section-header">Métricas Generales</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total de Tweets",
            value=f"{len(df_filtered):,}",
            delta=f"{len(df_filtered) - len(df):,}" if selected_users else None
        )

    with col2:
        st.metric(
            label="Usuarios Únicos",
            value=f"{df_filtered['username'].nunique():,}",
            delta=f"{df_filtered['username'].nunique() - df['username'].nunique():,}" if selected_users else None
        )

    with col3:
        avg_engagement = df_filtered[['retweet_count', 'like_count', 'reply_count']].sum().sum() / len(df_filtered)
        st.metric(
            label="Interacción Promedio",
            value=f"{avg_engagement:.1f}",
            delta=None
        )

    with col4:
        if 'word_count' in df_filtered.columns:
            avg_words = df_filtered['word_count'].mean()
            st.metric(
                label="Promedio Palabras/Tweet",
                value=f"{avg_words:.1f}",
                delta=None
            )

    # Visualization 1: Timeline Chart
    st.markdown('<h2 class="section-header">Análisis Temporal</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.plotly_chart(
            create_timeline_chart(df_filtered, date_range),
            use_container_width=True
        )

    # Visualization 2: User Analysis
    st.markdown('<h2 class="section-header">Análisis de Usuarios</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            create_top_users_chart(df_filtered),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            create_engagement_scatter(df_filtered),
            use_container_width=True
        )

    # Visualization 3: Content Analysis
    st.markdown('<h2 class="section-header">Análisis de Contenido</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            create_traffic_category_chart(df_filtered),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            create_engagement_trends(df_filtered),
            use_container_width=True
        )

    # Interactive filters demonstration
    st.markdown('<h2 class="section-header">Análisis Interactivo</h2>', unsafe_allow_html=True)

    # Category filter
    if 'traffic_category' in df_filtered.columns:
        categories = df_filtered['traffic_category'].unique()
        selected_categories = st.multiselect(
            "Filtrar por Categoría de Tráfico",
            options=categories,
            default=categories,
            key="category_filter"
        )

        # Aplicar filtro dinámico
        if selected_categories:
            df_cat_filtered = df_filtered[df_filtered['traffic_category'].isin(selected_categories)]
        else:
            df_cat_filtered = df_filtered

        # Mostrar métricas rápidas por categoría
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tweets en Categorías Seleccionadas", len(df_cat_filtered))
        with col2:
            avg_engagement = df_cat_filtered['engagement_total'].mean()
            st.metric("Interacción Promedio", f"{avg_engagement:.1f}")
        with col3:
            top_category = (
                df_cat_filtered['traffic_category'].mode().iloc[0]
                if len(df_cat_filtered) > 0 else "N/A"
            )
            st.metric("Categoría Principal", top_category)

        # --- NUEVA SECCIÓN: Visualizaciones enlazadas ---
        st.markdown('<h2 class="section-header">Análisis Espacial y Semántico</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                create_geo_heatmap(df_cat_filtered),
                use_container_width=True
            )

        with col2:
            st.markdown("#### Nube de Palabras de Tweets")
            create_wordcloud(df_cat_filtered)

    # Visualization 5: Modelos Predictivos
    st.markdown('<h2 class="section-header">Modelos Predictivos y Comparativa de Desempeño</h2>', unsafe_allow_html=True)

    # Entrenamiento de los modelos (cacheado para eficiencia)
    with st.spinner("Entrenando modelos predictivos..."):
        results_df, preds = train_models(df_filtered)

    if results_df is not None:
        # Mostrar tabla comparativa
        st.markdown("#### Desempeño General de los Modelos")
        st.dataframe(results_df, use_container_width=True)

        # Selector para comparar modelos
        selected_models = st.multiselect(
            "Seleccionar modelos para comparar",
            options=results_df['Modelo'].tolist(),
            default=results_df['Modelo'].tolist()[:2]
        )

        if selected_models:
            # Gráfico de barras comparativo
            fig = px.bar(
                results_df[results_df['Modelo'].isin(selected_models)],
                x='Modelo',
                y=['Accuracy', 'Precision', 'Recall', 'F1-score'],
                barmode='group',
                title='Comparativa de Desempeño entre Modelos'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Mostrar matrices de confusión
            st.markdown("#### Matrices de Confusión")
            cols = st.columns(len(selected_models))
            for i, name in enumerate(selected_models):
                with cols[i]:
                    y_true, y_pred = preds[name]
                    plot_confusion_matrix(y_true, y_pred, name)


    # Data table
    st.markdown('<h2 class="section-header">Explorador de Datos</h2>', unsafe_allow_html=True)

    if st.checkbox("Mostrar datos sin procesar"):
        display_columns = ['created_at', 'username', 'text_clean', 'retweet_count', 'like_count', 'traffic_category']
        available_columns = [col for col in display_columns if col in df_filtered.columns]
        st.dataframe(
            df_filtered[available_columns].head(100),
            use_container_width=True
        )

    # Footer
    st.markdown("---")
    st.markdown(
        "**Panel de Análisis de Tráfico GT** | Construido con Streamlit & Plotly | "
        f"Datos: {len(df)} tweets de {df['username'].nunique()} usuarios"
    )

if __name__ == "__main__":
    main()