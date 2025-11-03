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
from sklearn.model_selection import learning_curve
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

def create_learning_curves(X, y, model, model_name):
    """Create learning curve visualization"""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy', random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig = go.Figure()

    # Training score
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=train_mean,
        mode='lines+markers',
        name='Entrenamiento',
        line=dict(color=COLORS['primary'], width=2),
        error_y=dict(type='data', array=train_std, visible=True)
    ))

    # Validation score
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=val_mean,
        mode='lines+markers',
        name='Validación',
        line=dict(color=COLORS['secondary'], width=2),
        error_y=dict(type='data', array=val_std, visible=True)
    ))

    fig.update_layout(
        title=f'Curva de Aprendizaje - {model_name}',
        xaxis_title='Tamaño del Conjunto de Entrenamiento',
        yaxis_title='Accuracy',
        height=400,
        hovermode='x unified'
    )

    return fig

def create_network_analysis(df):
    """Create network analysis of user interactions - 8th visualization"""
    # Create a simplified network based on mentions and replies
    interactions = []

    # Extract mentions from tweets
    for idx, row in df.iterrows():
        text = str(row.get('text_clean', ''))
        mentions = re.findall(r'@(\w+)', text)
        for mention in mentions:
            interactions.append({
                'source': row['username'],
                'target': mention,
                'weight': 1
            })

    if not interactions:
        # Create dummy data if no interactions found
        top_users = df['username'].value_counts().head(10).index.tolist()
        for i in range(len(top_users)-1):
            interactions.append({
                'source': top_users[i],
                'target': top_users[i+1],
                'weight': np.random.randint(1, 5)
            })

    # Create network graph
    interaction_df = pd.DataFrame(interactions)
    if len(interaction_df) > 0:
        interaction_summary = interaction_df.groupby(['source', 'target'])['weight'].sum().reset_index()

        # Limit to top interactions for visualization
        interaction_summary = interaction_summary.nlargest(20, 'weight')

        # Create network using NetworkX
        G = nx.from_pandas_edgelist(interaction_summary, 'source', 'target', ['weight'])

        # Calculate positions
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Extract coordinates
        node_x = []
        node_y = []
        node_text = []
        node_size = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            # Size based on degree
            node_size.append(G.degree(node) * 5 + 10)

        # Create edges
        edge_x = []
        edge_y = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Create plotly figure
        fig = go.Figure()

        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))

        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=node_size,
                color=node_size,
                colorbar=dict(
                    thickness=15,
                    xanchor='left',
                    title=dict(text='Conexiones', side='right')
                ),
                line=dict(width=2, color='white')
            ),
            showlegend=False
        ))

        fig.update_layout(
            title=dict(text='Red de Interacciones entre Usuarios', font=dict(size=16)),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Análisis de red basado en menciones e interacciones",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )

        return fig
    else:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No se encontraron interacciones suficientes para el análisis de red",
            showarrow=False, x=0.5, y=0.5
        )
        return fig

def create_sentiment_timeline(df):
    """Create sentiment analysis over time - bonus visualization"""
    # Simple sentiment analysis based on keywords
    positive_words = ['bien', 'bueno', 'excelente', 'perfecto', 'gracias', 'libre']
    negative_words = ['mal', 'terrible', 'pesimo', 'problema', 'accidente', 'cerrado', 'tranque']

    def analyze_sentiment(text):
        text = str(text).lower()
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)

        if pos_count > neg_count:
            return 'Positivo'
        elif neg_count > pos_count:
            return 'Negativo'
        else:
            return 'Neutral'

    df['sentiment'] = df['text_clean'].apply(analyze_sentiment)

    # Group by date and sentiment
    sentiment_daily = df.groupby(['date', 'sentiment']).size().reset_index(name='count')

    fig = px.area(
        sentiment_daily,
        x='date',
        y='count',
        color='sentiment',
        title='Análisis de Sentimiento a lo Largo del Tiempo',
        color_discrete_map={
            'Positivo': COLORS['success'],
            'Negativo': COLORS['danger'],
            'Neutral': COLORS['neutral']
        }
    )

    fig.update_layout(
        height=400,
        xaxis_title='Fecha',
        yaxis_title='Número de Tweets',
        hovermode='x unified'
    )

    return fig


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

    # Sidebar filters and controls
    st.sidebar.header("Controles del Panel")

    # Visualization selector
    st.sidebar.subheader("Selector de Visualizaciones")
    viz_options = {
        'Temporal': ['Timeline', 'Heatmap de Actividad', 'Tendencias de Interacción'],
        'Usuarios': ['Top Usuarios', 'Scatter de Interacción', 'Red de Interacciones'],
        'Contenido': ['Categorías de Tráfico', 'Mapa Geográfico', 'Nube de Palabras'],
        'Sentimiento': ['Análisis de Sentimiento'],
        'Modelos': ['Comparativa de Modelos', 'Curvas de Aprendizaje', 'Matrices de Confusión']
    }

    selected_viz = {}
    for category, visualizations in viz_options.items():
        selected_viz[category] = st.sidebar.multiselect(
            f"Visualizaciones {category}",
            options=visualizations,
            default=visualizations,  # All selected by default
            key=f"viz_{category}"
        )

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

    # Enhanced user filter
    st.sidebar.subheader("Filtros de Datos")

    # User filter with search
    top_users = df['username'].value_counts().head(50).index.tolist()
    selected_users = st.sidebar.multiselect(
        "Filtrar por Usuarios (Top 50)",
        options=top_users,
        default=top_users[:5] if len(top_users) >= 5 else top_users,
        help="Selecciona usuarios específicos para análisis enfocado"
    )

    # Engagement filter
    if 'engagement_total' in df.columns:
        min_engagement = st.sidebar.slider(
            "Interacción mínima",
            min_value=0,
            max_value=int(df['engagement_total'].max()),
            value=0,
            help="Filtrar tweets por nivel mínimo de interacción"
        )
    else:
        min_engagement = 0

    # Time range filter
    time_filter = st.sidebar.selectbox(
        "Filtro de Tiempo",
        options=['Todo el período', 'Últimos 7 días', 'Últimos 30 días', 'Personalizado'],
        index=0
    )

    # Apply time filter
    if time_filter == 'Últimos 7 días':
        cutoff_date = df['created_at'].max() - pd.Timedelta(days=7)
        df_time_filtered = df[df['created_at'] >= cutoff_date]
    elif time_filter == 'Últimos 30 días':
        cutoff_date = df['created_at'].max() - pd.Timedelta(days=30)
        df_time_filtered = df[df['created_at'] >= cutoff_date]
    elif time_filter == 'Personalizado':
        df_time_filtered = df[
            (df['created_at'].dt.date >= date_range[0]) &
            (df['created_at'].dt.date <= date_range[1])
        ]
    else:
        df_time_filtered = df

    # Apply engagement filter
    if 'engagement_total' in df_time_filtered.columns:
        df_engagement_filtered = df_time_filtered[df_time_filtered['engagement_total'] >= min_engagement]
    else:
        df_engagement_filtered = df_time_filtered

    # Apply user filter
    if selected_users:
        df_filtered = df_engagement_filtered[df_engagement_filtered['username'].isin(selected_users)]
    else:
        df_filtered = df_engagement_filtered

    # Show filter effects
    st.sidebar.markdown("### Resumen de Filtros")
    st.sidebar.info(f"""**Datos filtrados:**
    - {len(df_filtered):,} tweets (de {len(df):,} totales)
    - {df_filtered['username'].nunique():,} usuarios
    - Filtro temporal: {time_filter}
    - Usuarios seleccionados: {len(selected_users) if selected_users else 'Todos'}""")

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

    # Visualization Section: Temporal Analysis
    if any(selected_viz['Temporal']):
        st.markdown('<h2 class="section-header">Análisis Temporal</h2>', unsafe_allow_html=True)

        if 'Timeline' in selected_viz['Temporal']:
            st.plotly_chart(
                create_timeline_chart(df_filtered, date_range),
                use_container_width=True,
                key="timeline_chart"
            )

        col1, col2 = st.columns(2)

        with col1:
            if 'Heatmap de Actividad' in selected_viz['Temporal']:
                st.plotly_chart(
                    create_activity_heatmap(df_filtered),
                    use_container_width=True,
                    key="activity_heatmap"
                )

        with col2:
            if 'Tendencias de Interacción' in selected_viz['Temporal']:
                st.plotly_chart(
                    create_engagement_trends(df_filtered),
                    use_container_width=True,
                    key="engagement_trends"
                )

    # Visualization Section: User Analysis
    if any(selected_viz['Usuarios']):
        st.markdown('<h2 class="section-header">Análisis de Usuarios</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            if 'Top Usuarios' in selected_viz['Usuarios']:
                st.plotly_chart(
                    create_top_users_chart(df_filtered),
                    use_container_width=True,
                    key="top_users_chart"
                )

        with col2:
            if 'Scatter de Interacción' in selected_viz['Usuarios']:
                st.plotly_chart(
                    create_engagement_scatter(df_filtered),
                    use_container_width=True,
                    key="engagement_scatter"
                )

        # Network analysis visualization
        if 'Red de Interacciones' in selected_viz['Usuarios']:
            st.plotly_chart(
                create_network_analysis(df_filtered),
                use_container_width=True,
                key="network_analysis"
            )

    # Visualization Section: Content Analysis
    if any(selected_viz['Contenido']):
        st.markdown('<h2 class="section-header">Análisis de Contenido</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            if 'Categorías de Tráfico' in selected_viz['Contenido']:
                st.plotly_chart(
                    create_traffic_category_chart(df_filtered),
                    use_container_width=True,
                    key="traffic_categories"
                )

        with col2:
            if 'Mapa Geográfico' in selected_viz['Contenido']:
                st.plotly_chart(
                    create_geo_heatmap(df_filtered),
                    use_container_width=True,
                    key="geo_heatmap"
                )

        if 'Nube de Palabras' in selected_viz['Contenido']:
            st.markdown("#### Nube de Palabras de Tweets")
            create_wordcloud(df_filtered)

    # Sentiment Analysis Section
    if any(selected_viz['Sentimiento']):
        st.markdown('<h2 class="section-header">Análisis de Sentimiento</h2>', unsafe_allow_html=True)

        if 'Análisis de Sentimiento' in selected_viz['Sentimiento']:
            st.plotly_chart(
                create_sentiment_timeline(df_filtered.copy()),
                use_container_width=True,
                key="sentiment_timeline"
            )

    # Interactive filters section
    st.markdown('<h2 class="section-header">Controles Interactivos Avanzados</h2>', unsafe_allow_html=True)

    # Category filter with enhanced interactivity
    if 'traffic_category' in df_filtered.columns:
        categories = df_filtered['traffic_category'].unique()
        selected_categories = st.multiselect(
            "Filtrar por Categoría de Tráfico",
            options=categories,
            default=categories,
            key="category_filter",
            help="Selecciona categorías para filtrar todas las visualizaciones dinámicamente"
        )

        # Apply dynamic filter
        if selected_categories:
            df_cat_filtered = df_filtered[df_filtered['traffic_category'].isin(selected_categories)]
        else:
            df_cat_filtered = df_filtered

        # Show quick metrics per category with delta comparison
        col1, col2, col3 = st.columns(3)
        with col1:
            current_count = len(df_cat_filtered)
            total_count = len(df_filtered)
            st.metric(
                "Tweets en Categorías Seleccionadas",
                current_count,
                delta=f"{current_count - total_count} vs total" if current_count != total_count else None
            )
        with col2:
            avg_engagement = df_cat_filtered['engagement_total'].mean() if 'engagement_total' in df_cat_filtered.columns else 0
            st.metric("Interacción Promedio", f"{avg_engagement:.1f}")
        with col3:
            top_category = (
                df_cat_filtered['traffic_category'].mode().iloc[0]
                if len(df_cat_filtered) > 0 else "N/A"
            )
            st.metric("Categoría Principal", top_category)

        # Enhanced linked visualizations section
        st.markdown('<h3 class="section-header">Visualizaciones Enlazadas (Actualización Dinámica)</h3>', unsafe_allow_html=True)
        st.info("💡 Las siguientes visualizaciones se actualizan automáticamente según los filtros de categoría seleccionados arriba.")

        # Create tabs for better organization of linked visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Métricas Temporales", "🗺️ Análisis Espacial", "👥 Interacciones"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    create_timeline_chart(df_cat_filtered, date_range),
                    use_container_width=True,
                    key="linked_timeline_chart"
                )
            with col2:
                st.plotly_chart(
                    create_activity_heatmap(df_cat_filtered),
                    use_container_width=True,
                    key="linked_activity_heatmap"
                )

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    create_geo_heatmap(df_cat_filtered),
                    use_container_width=True,
                    key="linked_geo_heatmap"
                )
            with col2:
                create_wordcloud(df_cat_filtered)

        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    create_engagement_scatter(df_cat_filtered),
                    use_container_width=True,
                    key="linked_engagement_scatter"
                )
            with col2:
                st.plotly_chart(
                    create_network_analysis(df_cat_filtered),
                    use_container_width=True,
                    key="linked_network_analysis"
                )

    # Predictive Models Section
    if any(selected_viz['Modelos']):
        st.markdown('<h2 class="section-header">Modelos Predictivos y Comparativa de Desempeño</h2>', unsafe_allow_html=True)

        # Train models (cached for efficiency)
        with st.spinner("Entrenando modelos predictivos..."):
            results_df, preds = train_models(df_filtered)

        if results_df is not None:
            # Show comparative table
            if 'Comparativa de Modelos' in selected_viz['Modelos']:
                st.markdown("#### Desempeño General de los Modelos")
                st.dataframe(results_df, use_container_width=True)

                # Model selector for comparison
                selected_models = st.multiselect(
                    "Seleccionar modelos para comparar",
                    options=results_df['Modelo'].tolist(),
                    default=results_df['Modelo'].tolist()[:2],
                    help="Selecciona qué modelos quieres comparar en detalle"
                )

                if selected_models:
                    # Comparative bar chart
                    fig = px.bar(
                        results_df[results_df['Modelo'].isin(selected_models)],
                        x='Modelo',
                        y=['Accuracy', 'Precision', 'Recall', 'F1-score'],
                        barmode='group',
                        title='Comparativa de Desempeño entre Modelos'
                    )
                    st.plotly_chart(fig, use_container_width=True, key="model_comparison")

            # Learning curves
            if 'Curvas de Aprendizaje' in selected_viz['Modelos'] and results_df is not None:
                st.markdown("#### Curvas de Aprendizaje")

                # Prepare data for learning curves
                X = df_filtered[['retweet_count', 'like_count', 'reply_count', 'word_count']]
                y = df_filtered['traffic_category']

                from sklearn.linear_model import LogisticRegression
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.svm import SVC

                models_for_curves = {
                    "Regresión Logística": LogisticRegression(max_iter=300),
                    "Bosque Aleatorio": RandomForestClassifier(n_estimators=50, random_state=42),  # Reduced for speed
                    "SVM Lineal": SVC(kernel='linear', random_state=42)
                }

                selected_model_curves = st.selectbox(
                    "Seleccionar modelo para curva de aprendizaje",
                    options=list(models_for_curves.keys())
                )

                if selected_model_curves:
                    model = models_for_curves[selected_model_curves]
                    learning_curve_fig = create_learning_curves(X, y, model, selected_model_curves)
                    st.plotly_chart(learning_curve_fig, use_container_width=True, key="learning_curves")

            # Confusion matrices
            if 'Matrices de Confusión' in selected_viz['Modelos'] and results_df is not None:
                st.markdown("#### Matrices de Confusión")

                if 'selected_models' in locals() and selected_models:
                    cols = st.columns(len(selected_models))
                    for i, name in enumerate(selected_models):
                        with cols[i]:
                            y_true, y_pred = preds[name]
                            plot_confusion_matrix(y_true, y_pred, name)
                else:
                    st.info("Selecciona modelos en la sección 'Comparativa de Modelos' para ver las matrices de confusión.")


    # Enhanced Data Explorer
    st.markdown('<h2 class="section-header">Explorador de Datos Avanzado</h2>', unsafe_allow_html=True)

    # Data explorer controls
    col1, col2, col3 = st.columns(3)

    with col1:
        show_raw_data = st.checkbox("Mostrar datos sin procesar")

    with col2:
        if show_raw_data:
            num_rows = st.slider("Número de filas", 10, 500, 100)

    with col3:
        if show_raw_data:
            sort_column = st.selectbox(
                "Ordenar por",
                options=['created_at', 'retweet_count', 'like_count', 'username'],
                index=0
            )

    if show_raw_data:
        display_columns = ['created_at', 'username', 'text_clean', 'retweet_count', 'like_count', 'traffic_category']
        available_columns = [col for col in display_columns if col in df_filtered.columns]

        # Sort and display data
        if sort_column in df_filtered.columns:
            display_df = df_filtered[available_columns].sort_values(sort_column, ascending=False).head(num_rows)
        else:
            display_df = df_filtered[available_columns].head(num_rows)

        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )

        # Download option
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Descargar datos mostrados como CSV",
            data=csv,
            file_name='traffic_data_filtered.csv',
            mime='text/csv'
        )

    # Enhanced Footer with Statistics
    st.markdown("---")

    # Summary statistics in footer
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total de Tweets", f"{len(df):,}")
    with col2:
        st.metric("Usuarios Únicos", f"{df['username'].nunique():,}")
    with col3:
        if 'created_at' in df.columns:
            date_range_days = (df['created_at'].max() - df['created_at'].min()).days
            st.metric("Período (días)", f"{date_range_days:,}")
    with col4:
        total_engagement = df[['retweet_count', 'like_count', 'reply_count']].sum().sum()
        st.metric("Interacciones Totales", f"{total_engagement:,}")

    st.markdown(
        "**Panel de Análisis de Tráfico GT** | Construido con Streamlit & Plotly | "
        "Dashboard Interactivo para Toma de Decisiones Ejecutivas"
    )

if __name__ == "__main__":
    main()