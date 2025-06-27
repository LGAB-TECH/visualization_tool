import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import io
import tempfile
import os
from PIL import Image
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

st.set_page_config(
    page_title="Smart Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown(
    """
    <style>
    .main { background-color: #f5f7fa; }
    .stApp header {background: linear-gradient(90deg,#1e3c72,#2a5298); color:white;}
    .stButton>button { background-color: #2a5298; color: white; font-weight: bold; border-radius: 8px;}
    .stButton>button:hover { background-color: #1e3c72; color: #fff;}
    .stSelectbox > div { background: #eef3fb;}
    .user-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #2a5298;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="user-info">
        <p style="margin:0; color:#1e3c72; font-size:0.9em;">
            <strong>🕒 Current Time (UTC):</strong> 2025-06-26 15:30:18<br>
            <strong>👤 User:</strong> LGAB-TECH
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 style='color:#2a5298; font-size:2.8rem; font-weight:700;'>📊 Smart Data Cleaner + Analyzer</h1>",
    unsafe_allow_html=True
)

# ---- Session state ----
for k, v in {
    "df": None, "charts": {},  "df_original": None, "show_bar_plot": False
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def auto_detect_and_convert_dtypes(df, cat_unique_thresh=20, cat_percent_thresh=0.1):
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_categorical_dtype(series):
            continue
        coerced = pd.to_numeric(series, errors='coerce')
        n_nonnull = series.notna().sum()
        n_numeric = coerced.notna().sum()
        if n_nonnull > 0 and n_numeric / n_nonnull > 0.9:
            if (coerced.dropna() % 1 == 0).all():
                if coerced.isna().sum() == 0:
                    df[col] = coerced.astype(int)
                else:
                    df[col] = coerced.astype(float)
            else:
                df[col] = coerced.astype(float)
            continue
        coerced = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
        n_datetime = coerced.notna().sum()
        if n_nonnull > 0 and n_datetime / n_nonnull > 0.9:
            df[col] = coerced
            continue
        n_unique = series.nunique(dropna=True)
        if len(df) > 0 and (n_unique <= cat_unique_thresh or n_unique/len(df) <= cat_percent_thresh):
            df[col] = series.astype('category')
            continue
    return df

def clean_and_fill_all_but_allna(df):
    """Clean dataframe by dropping only all-NA columns, converting numerics, filling NA with mean/mode."""
    if df is None:
        return None, [], []
    df = df.copy()
    df.columns = df.columns.str.strip()
    # Drop columns that are all NA
    allna_cols = df.columns[df.isna().all()].tolist()
    df = df.drop(columns=allna_cols)
    cleaned_columns = []
    # Try to convert all columns to numeric if possible (coerce errors to NaN)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isnull().any():
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
                cleaned_columns.append(col)
        elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "unknown"
                df[col] = df[col].fillna(mode_val)
                cleaned_columns.append(col)
            df[col] = df[col].astype(str).str.strip().str.lower()
    return df, cleaned_columns, allna_cols

def get_numeric_columns(df):
    return df.select_dtypes(include=['int64', 'float64']).columns.tolist()

def get_categorical_columns(df):
    return df.select_dtypes(include=['category', 'object']).columns.tolist()

def save_plotly_as_png(fig):
    """Save the plot as PNG using either Kaleido or Selenium screenshot fallback"""
    try:
        # First try Kaleido if available
        try:
            import kaleido
            img_bytes = pio.to_image(
                fig,
                format='png',
                width=1200,
                height=800,
                scale=2,
                engine="kaleido"
            )
            return img_bytes
        except ImportError:
            pass
       
        # Fall back to Selenium screenshot
        return take_selenium_screenshot()
   
    except Exception as e:
        st.error(f"Export failed: {str(e)}")
        return None
   
    except Exception as e:
        st.error(f"Failed to export identical plot: {str(e)}")
        return None

def take_selenium_screenshot():
    """Capture the current Streamlit plot using Selenium"""
    try:
        # Configure Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--force-device-scale-factor=2")
        chrome_options.add_argument("--hide-scrollbars")
       
        # Initialize the driver
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
       
        try:
            # Capture the current Streamlit app
            driver.get("http://localhost:8501")
           
            # Wait for plot to render
            driver.implicitly_wait(5)
           
            # Find the most recent Plotly chart
            plot_element = driver.find_element("css selector", ".stPlotlyChart:last-child")
           
            # Add white background and padding
            driver.execute_script("""
                arguments[0].style.background = 'white';
                arguments[0].style.padding = '20px';
            """, plot_element)
           
            return plot_element.screenshot_as_png
           
        finally:
            driver.quit()
           
    except Exception as e:
        st.error(f"Selenium screenshot failed: {str(e)}")
        return None

with st.sidebar:
    st.markdown("## File Selection Mode")
    file_mode = st.radio(
        "Choose analysis mode:",
        ("Single CSV Analysis", "Multiple CSV Merge & Analysis"),
        help="Select if you want to analyze a single CSV or merge multiple CSVs before analysis."
    )
   
df = None

# ---- File Upload & Merge Logic with Session State ----
if file_mode == "Multiple CSV Merge & Analysis":
    st.markdown("### 📁 Combine Multiple CSV Files (by Primary Key)", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "📂 <span style='font-size:19px;color:#2a5298;'>Upload one or more CSV files (for merging)</span>",
        type=["csv"],
        help="Upload multiple CSVs to merge them by primary key",
        accept_multiple_files=True,
        key="multi_csv_uploader"
    )

    if uploaded_files and len(uploaded_files) > 1:
        st.info("You uploaded multiple CSVs. Select primary key(s) to merge them.")
        dfs = []
        file_names = []
        for i, file in enumerate(uploaded_files):
            try:
                temp_df = pd.read_csv(file)
                temp_df.columns = temp_df.columns.str.strip()
                dfs.append(temp_df)
                file_names.append(file.name)
            except Exception as e:
                st.error(f"❌ Error reading {file.name}: {e}")

        st.markdown("#### Columns in each file:")
        cols_dict = {file_names[i]: dfs[i].columns.tolist() for i in range(len(dfs))}
        for fname, cols in cols_dict.items():
            st.write(f"**{fname}**: {cols}")
        common_cols = list(set.intersection(*(set(cols) for cols in cols_dict.values())))
        if common_cols:
            selected_keys = st.multiselect("🔑 Select primary key column(s) for merging:", common_cols, default=[common_cols[0]])
            how = st.selectbox("➕ Merge type:", ["inner", "outer", "left", "right"], help="""
                - inner: Only rows with matching keys in all files
                - outer: All rows, fill missing with NaN
                - left: All rows from the first file, matching from others
                - right: All rows from the last file, matching from others
            """)
            if st.button("🔗 Merge Files"):
                merged_df = dfs[0]
                for d in dfs[1:]:
                    merged_df = pd.merge(merged_df, d, on=selected_keys, how=how, suffixes=('', '_dup'))
                st.success(f"✅ Merged {len(uploaded_files)} files on {selected_keys} ({how} join).")
                st.dataframe(merged_df.head(10), use_container_width=True)
                st.session_state['merged_df'] = merged_df.copy()
        else:
            st.warning("⚠️ No common columns found across all CSVs. Cannot merge.")

    if 'merged_df' in st.session_state:
        df = st.session_state['merged_df']

elif file_mode == "Single CSV Analysis":
    uploaded_file = st.file_uploader(
        "📂 <span style='font-size:19px;color:#2a5298;'>Upload your CSV or Excel file</span>",
        type=["csv", "xlsx"],
        help="Supported formats: CSV, XLSX",
        key="single_file_uploader"
    )
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.session_state['merged_df'] = df.copy()
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            df = None
    if 'merged_df' in st.session_state:
        df = st.session_state['merged_df']

tab_labels = [
    "📋 Data Overview",
    "📈 Correlation & MI Analysis",
    "📊 Visualization"
]
if "tab_radio" not in st.session_state:
    st.session_state["tab_radio"] = tab_labels[0]
selected_tab = st.radio("Navigation", tab_labels, horizontal=True, key="tab_radio")

def goto_bar_plot_tab():
    st.session_state["tab_radio"] = tab_labels[2]
    st.session_state["show_bar_plot"] = True
    st.rerun()  # Use st.rerun() instead of st.experimental_rerun()

if df is not None:
    df_clean, cleaned_columns, allna_cols = clean_and_fill_all_but_allna(df)
    df = df_clean

    if selected_tab == tab_labels[0]:
        st.markdown("<h3 style='color:#1e3c72;'>📋 Dataset Overview</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Dataset Shape**")
            st.write(f"{df.shape[0]} rows × {df.shape[1]} columns")
        with col2:
            st.markdown("**Missing Values After Cleaning**")
            st.write("No missing values remaining" if df.isnull().sum().sum() == 0 else
                    f"{df.isnull().sum().sum()} missing values remaining")
        st.markdown("<h4 style='color:#2a5298;margin-top:20px;'>📝 Data Types</h4>", unsafe_allow_html=True)
        dtypes_df = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': [str(df[col].dtype) for col in df.columns],
            'Sample Values': [str(df[col].head(3).tolist()) for col in df.columns],
            'Missing Cleaned': [("Yes" if col in cleaned_columns else "No") for col in df.columns]
        })
        st.dataframe(
            dtypes_df,
            use_container_width=True,
            height=400
        )
        if allna_cols:
            st.markdown(
                f"<span style='color:#FF9900;'>Dropped columns with all missing values: <b>{', '.join(allna_cols)}</b></span>",
                unsafe_allow_html=True
            )
        st.markdown("<h4 style='color:#2a5298;margin-top:20px;'>🔍 Data Preview</h4>", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

    elif selected_tab == tab_labels[1]:
        st.markdown("<h3 style='color:#1e3c72;'>🔥 Correlation Analysis</h3>", unsafe_allow_html=True)
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            correlation_type = st.selectbox(
                "📊 Select Correlation Method:",
                ["Pearson", "Spearman", "Kendall"],
                help="Pearson: Linear correlation\nSpearman: Monotonic correlation\nKendall: Ordinal correlation"
            )
            show_vals = st.checkbox("🔢 Show correlation values", value=True)
            if correlation_type == "Spearman":
                corr = numeric_df.corr(method='spearman')
            elif correlation_type == "Kendall":
                corr = numeric_df.corr(method='kendall')
            else:
                corr = numeric_df.corr(method='pearson')
            corr = corr.fillna(0)
            n_vars = len(corr.columns)
            text_size = max(8, min(10, int(400 / n_vars)))
            hover_text = np.round(corr, 2).astype(str)
            fig = go.Figure(data=go.Heatmap(
                z=corr,
                x=corr.columns,
                y=corr.columns,
                zmin=-1,
                zmax=1,
                text=hover_text,
                texttemplate="%{text}" if show_vals else None,
                textfont={"size": text_size},
                hoverongaps=False,
                hovertemplate="<b>x: %{x}</b><br><b>y: %{y}</b><br><b>correlation: %{z:.2f}</b><extra></extra>",
                colorscale="RdBu_r",
                showscale=True
            ))
            fig.update_layout(
                title=dict(
                    text=f"Interactive {correlation_type} Correlation Matrix",
                    x=0.5,
                    xanchor='center',
                    font=dict(size=20, color='#1e3c72')
                ),
                width=max(500, min(800, 100 * n_vars)),
                height=max(500, min(800, 100 * n_vars)),
                xaxis=dict(
                    tickangle=45,
                    tickfont=dict(size=text_size, color="black"),
                    showgrid=False
                ),
                yaxis=dict(
                    tickfont=dict(size=text_size, color="black"),
                    showgrid=False,
                    autorange='reversed'
                ),
                margin=dict(
                    l=100,
                    r=50,
                    t=120,
                    b=100
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            if cleaned_columns:
                st.markdown(
                    "<span style='color:#FF9900;'>Columns that had missing values filled and are included in the correlation matrix: <b>{}</b></span>".format(", ".join([col for col in numeric_df.columns if col in cleaned_columns])),
                    unsafe_allow_html=True
                )
            if allna_cols:
                st.markdown(
                    "<span style='color:#FF3333;'>Columns dropped from correlation matrix because they were all NA: <b>{}</b></span>".format(", ".join(allna_cols)),
                    unsafe_allow_html=True
                )

            st.markdown("<h4 style='color:#2a5298;margin-top:30px;'>🎯 Mutual Information with Target</h4>", unsafe_allow_html=True)
            all_cols = df.columns.tolist()
            target_col = st.selectbox("Select target column:", all_cols)
            if target_col:
                features = [col for col in df.columns if col != target_col]
                target_data = df[target_col]
                if pd.api.types.is_numeric_dtype(target_data):
                    task_type = st.radio("Is your target variable a classification or regression task?",
                                       ["Regression", "Classification"], index=0)
                else:
                    task_type = "Classification"
                X = df[features].copy()
                for col in X.select_dtypes(include=["object", "category"]).columns:
                    X[col], _ = X[col].factorize()
                y = target_data
                if task_type == "Classification":
                    y, _ = pd.factorize(y)
                try:
                    if task_type == "Regression":
                        mi = mutual_info_regression(X, y, random_state=0)
                    else:
                        mi = mutual_info_classif(X, y, random_state=0)
                except Exception as e:
                    st.error(f"Mutual information calculation failed: {e}")
                    mi = None
                if mi is not None:
                    mi_series = pd.Series(mi, index=features)
                    mi_sorted = mi_series.sort_values(ascending=False)
                    fig_mi = go.Figure()
                    fig_mi.add_trace(go.Bar(
                        x=mi_sorted.values,
                        y=mi_sorted.index,
                        orientation='h',
                        marker=dict(
                            color=mi_sorted.values,
                            colorscale='Blues',
                            cmin=0,
                            cmax=mi_sorted.max() if mi_sorted.max() > 0 else 1,
                            line=dict(width=1, color='#333333')
                        ),
                        hovertemplate='<b>%{y}</b><br>Mutual Info: %{x:.3f}<extra></extra>'
                    ))
                    fig_mi.update_layout(
                        title=dict(
                            text=f"Mutual Information of Variables with {target_col}",
                            x=0.5,
                            xanchor='center',
                            font=dict(
                                size=20,
                                color='#1e3c72',
                                family='Arial, sans-serif'
                            )
                        ),
                        xaxis_title=dict(
                            text="Mutual Information",
                            font=dict(size=14, color='#2a5298')
                        ),
                        yaxis_title=dict(
                            text="Variables",
                            font=dict(size=14, color='#2a5298')
                        ),
                        width=900,
                        height=max(400, len(mi_sorted) * 30),
                        margin=dict(l=200, r=40, t=80, b=40),
                        xaxis=dict(
                            zeroline=True,
                            zerolinecolor='#333333',
                            zerolinewidth=1,
                            tickfont=dict(size=12),
                            gridcolor='#E5E5E5',
                            showgrid=True,
                            gridwidth=1
                        ),
                        yaxis=dict(
                            autorange="reversed",
                            tickfont=dict(size=12),
                            gridcolor='#E5E5E5'
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        showlegend=False,
                        shapes=[
                            dict(
                                type='rect',
                                xref='paper',
                                yref='paper',
                                x0=0,
                                y0=0,
                                x1=1,
                                y1=1,
                                line=dict(
                                    color='#2a5298',
                                    width=2
                                ),
                                fillcolor='rgba(0,0,0,0)'
                            )
                        ]
                    )
                    st.plotly_chart(fig_mi, use_container_width=True)
                    st.markdown("<h4 style='color:#2a5298;margin-top:20px;'>📊 Mutual Information Details</h4>", unsafe_allow_html=True)
                    st.dataframe(
                        pd.DataFrame({
                            "Feature": mi_sorted.index,
                            "Mutual Information": mi_sorted.values.round(4)
                        }),
                        use_container_width=True,
                        height=350
                    )
        else:
            st.warning("⚠️ No numeric columns found in the dataset!")

    elif selected_tab == tab_labels[2]:
        st.markdown("<h3 style='color:#1e3c72;'>📊 Visualization</h3>", unsafe_allow_html=True)
        st.write("Create custom plots (Bar, Smooth Area, Line, Scatter, and Combinations)")
        categorical_cols = get_categorical_columns(df)
        numeric_cols = get_numeric_columns(df)

        plot_type = st.selectbox(
            "📊 Plot Type:",
            [
                "Grouped Bar",
                "Stacked Bar",
                "Grouped Bar with Separators",
                "Side by Side Bar",
                "Smooth Area Plot",
                "Bell Curve Area Plot",
                "Simple Line Plot",
                "Grouped Line Plot",
                "Scatter Plot"
            ]
        )

        if plot_type == "Bell Curve Area Plot":
            bell_col = st.selectbox("Select numeric column for bell curve (KDE):", numeric_cols)
            show_hist = st.checkbox("Show histogram overlay", value=True)
            xaxis_font_size = st.slider("Font Size for X-axis Values", min_value=6, max_value=32, value=12, key="bell_x_font_size")
            yaxis_font_size = st.slider("Font Size for Y-axis Values", min_value=6, max_value=32, value=12, key="bell_y_font_size")
            fig = go.Figure()
            if bell_col:
                bell_data = df[bell_col].dropna()
                if len(bell_data) < 2:
                    st.warning("Not enough data to plot a distribution curve.")
                else:
                    from scipy.stats import gaussian_kde
                    x_grid = np.linspace(bell_data.min(), bell_data.max(), 200)
                    kde = gaussian_kde(bell_data)
                    y_density = kde(x_grid)
                    if show_hist:
                        fig.add_trace(go.Histogram(
                            x=bell_data,
                            histnorm='probability density',
                            opacity=0.3,
                            name="Histogram",
                            marker_color="#a5c8e1"
                        ))
                    fig.add_trace(go.Scatter(
                        x=x_grid,
                        y=y_density,
                        mode='lines',
                        fill='tozeroy',
                        name="KDE",
                        line=dict(color="#1e3c72", width=3),
                    ))
                    fig.update_layout(
                        title=f"Bell Curve (KDE) Area Plot for {bell_col}",
                        xaxis_title=bell_col,
                        yaxis_title="Density",
                        yaxis=dict(showgrid=False, tickfont=dict(size=yaxis_font_size)),
                        xaxis=dict(tickfont=dict(size=xaxis_font_size)),
                        margin=dict(t=100, b=100)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state.charts["Visualization Chart"] = fig
        elif plot_type == "Smooth Area Plot":
            smooth_col = st.selectbox("Select numeric column for smooth area plot:", numeric_cols)
            group_col = st.selectbox("Group by (optional):", ["None"] + categorical_cols)
            show_hist = st.checkbox("Show histogram overlay", value=False)
            xaxis_font_size = st.slider("Font Size for X-axis Values", min_value=6, max_value=32, value=12, key="smooth_x_font_size")
            yaxis_font_size = st.slider("Font Size for Y-axis Values", min_value=6, max_value=32, value=12, key="smooth_y_font_size")
            fig = go.Figure()
            if smooth_col:
                if group_col == "None":
                    data = df[smooth_col].dropna()
                    if len(data) < 2:
                        st.warning("Not enough data to plot a smooth area curve.")
                    else:
                        from scipy.stats import gaussian_kde
                        x_grid = np.linspace(data.min(), data.max(), 200)
                        kde = gaussian_kde(data)
                        y_density = kde(x_grid)
                        if show_hist:
                            fig.add_trace(go.Histogram(
                                x=data,
                                histnorm='probability density',
                                opacity=0.3,
                                name="Histogram",
                                marker_color="#a5c8e1"
                            ))
                        fig.add_trace(go.Scatter(
                            x=x_grid,
                            y=y_density,
                            mode='lines',
                            fill='tozeroy',
                            name="Smooth Area",
                            line=dict(color="#1e3c72", width=3),
                        ))
                else:
                    for i, group in enumerate(df[group_col].dropna().unique()):
                        group_data = df[df[group_col] == group][smooth_col].dropna()
                        if len(group_data) < 2:
                            continue
                        from scipy.stats import gaussian_kde
                        x_grid = np.linspace(group_data.min(), group_data.max(), 200)
                        kde = gaussian_kde(group_data)
                        y_density = kde(x_grid)
                        fig.add_trace(go.Scatter(
                            x=x_grid,
                            y=y_density,
                            mode='lines',
                            fill='tozeroy',
                            name=str(group),
                            line=dict(width=3),
                        ))
            fig.update_layout(
                title=f"Smooth Area Plot for {smooth_col}" + (f" by {group_col}" if group_col != "None" else ""),
                xaxis_title=smooth_col,
                yaxis_title="Density",
                yaxis=dict(showgrid=False, tickfont=dict(size=yaxis_font_size)),
                xaxis=dict(tickfont=dict(size=xaxis_font_size)),
                margin=dict(t=100, b=100)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.charts["Visualization Chart"] = fig
        else:
            # Rest of the visualization code for other plot types
            x_axis = st.selectbox("🧭 Select X-axis (categorical or numeric):", categorical_cols + numeric_cols)
            x_axis_values = df[x_axis].dropna().unique().tolist() if x_axis in df.columns else []
            selected_x_values = st.multiselect(
                f"Select {x_axis} values to display:",
                options=x_axis_values,
                default=x_axis_values
            ) if x_axis in categorical_cols else []
            filtered_df = df[df[x_axis].isin(selected_x_values)] if x_axis in categorical_cols else df

            y_axis = st.selectbox("📏 Select Y-axis (numeric):", numeric_cols)
            agg_func = None
            stack_column = None
            group_column = None
            bar_color = "#1e3c72"
            show_average = False

            with st.expander("Customize Plot Labels & Title", expanded=False):
                custom_title = st.text_input("Plot Title (leave blank for auto)", key="custom_title")
                custom_x_label = st.text_input("X-axis Label (leave blank for column name)", key="custom_x_label")
                custom_y_label = st.text_input("Y-axis Label (leave blank for column name)", key="custom_y_label")
                custom_legend_names = st.text_area("Legend Names (comma-separated, leave blank for auto)", key="custom_legend_names")
                legend_names_list = [s.strip() for s in custom_legend_names.split(",")] if custom_legend_names else None
                xaxis_font_size = st.slider("Font Size for X-axis Values", min_value=6, max_value=32, value=12, key="x_font_size")
                yaxis_font_size = st.slider("Font Size for Y-axis Values", min_value=6, max_value=32, value=12, key="y_font_size")

            if plot_type in ["Grouped Bar", "Stacked Bar", "Side by Side Bar"]:
                agg_func = st.selectbox(
                    "📐 Aggregation Function:",
                    ["Sum", "Mean", "Median", "Count", "Standard Deviation", "Minimum", "Maximum"]
                )
            if plot_type in ["Grouped Bar", "Stacked Bar", "Simple Line Plot", "Scatter Plot", "Side by Side Bar"]:
                show_average = st.checkbox("Show Average Line", value=True)
            if plot_type == "Stacked Bar":
                stack_column = st.selectbox(
                    "🔢 Select Stacking Column:",
                    [col for col in categorical_cols if col != x_axis]
                )
            if plot_type == "Grouped Bar with Separators":
                group_column = st.selectbox(
                    "🏷️ Select Group Column:",
                    [col for col in categorical_cols if col != x_axis]
                )
                bar_color = st.color_picker("🎨 Select Bar Color:", "#FF0000")
            if plot_type == "Grouped Line Plot":
                group_column = st.selectbox(
                    "🏷️ Select Group Column (for Line):",
                    [col for col in categorical_cols if col != x_axis]
                )
            if plot_type == "Scatter Plot":
                scatter_x_axis = st.selectbox("Scatter X-axis (numeric):", numeric_cols)
                scatter_y_axis = st.selectbox("Scatter Y-axis (numeric):", [col for col in numeric_cols if col != scatter_x_axis])
                scatter_color = st.selectbox("Color by (optional categorical):", ["None"] + categorical_cols)
                scatter_size = st.selectbox("Size by (optional numeric):", ["None"] + numeric_cols)

            if plot_type == "Side by Side Bar":
                group_column = st.selectbox(
                    "🏷️ Select Group Column (for Side by Side):",
                    [col for col in categorical_cols if col != x_axis]
                )
                side_bar_colors = []
                if group_column:
                    unique_groups = filtered_df[group_column].unique()
                    for i, g in enumerate(unique_groups):
                        color = st.color_picker(f"Color for {g}:", px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)], key=f"side_color_{g}")
                        side_bar_colors.append(color)

            fig = go.Figure()
            func_map = {
                "Sum": "sum",
                "Mean": "mean",
                "Median": "median",
                "Count": "count",
                "Standard Deviation": "std",
                "Minimum": "min",
                "Maximum": "max"
            }
            title = None

            if plot_type == "Grouped Bar":
                group_df = filtered_df[[x_axis, y_axis]].dropna().groupby(x_axis)
                y_data = getattr(group_df, func_map[agg_func])().reset_index()
                n_bars = len(y_data)
                plot_width = max(500, min(80 * n_bars, 1600))
                plot_height = 500 if n_bars < 10 else min(100 + 30 * n_bars, 800)
                legend_name = legend_names_list[0] if legend_names_list else y_axis
                fig = px.bar(
                    y_data, x=x_axis, y=y_axis,
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                    opacity=0.9,
                    text_auto=True
                )
                if show_average and not y_data.empty:
                    avg = y_data[y_axis].mean()
                    fig.add_hline(y=avg, line_dash='dash', line_color='red', annotation_text=f"Avg: {avg:.2f}", annotation_position="top right")
                title = f"{agg_func} of {y_axis} by {x_axis}"

            elif plot_type == "Stacked Bar":
                group_df = filtered_df[[x_axis, stack_column, y_axis]].dropna()
                plot_df = getattr(group_df.groupby([x_axis, stack_column])[y_axis], func_map[agg_func])().reset_index()
                n_x = plot_df[x_axis].nunique()
                n_groups = plot_df[stack_column].nunique()
                plot_width = max(600, min(90 * n_x, 1800))
                plot_height = 500 if n_x < 10 else min(140 + 38 * n_x, 900)
                fig = px.bar(
                    plot_df, x=x_axis, y=y_axis, color=stack_column,
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                    opacity=0.9,
                    text_auto=True
                )
                if show_average and not plot_df.empty:
                    avg = plot_df[y_axis].mean()
                    fig.add_hline(y=avg, line_dash='dash', line_color='red', annotation_text=f"Avg: {avg:.2f}", annotation_position="top right")
                title = f"{agg_func} of {y_axis} by {x_axis} and {stack_column}"

            elif plot_type == "Grouped Bar with Separators":
                plot_data = filtered_df[[x_axis, group_column, y_axis]].dropna()
                means_df = plot_data.groupby([x_axis, group_column])[y_axis].mean().reset_index()
                group_vals = plot_data[group_column].unique()
                x_vals = plot_data[x_axis].unique()
                bar_width = 0.8 / len(group_vals) if len(group_vals) else 0.8
                for i, group in enumerate(group_vals):
                    y_vals = []
                    for x in x_vals:
                        val = plot_data[(plot_data[x_axis] == x) & (plot_data[group_column] == group)][y_axis]
                        y_vals.append(val.mean() if len(val) > 0 else 0)
                    legend_name = (
                        legend_names_list[i] if legend_names_list and i < len(legend_names_list)
                        else str(group)
                    )
                    fig.add_trace(go.Bar(
                        x=x_vals,
                        y=y_vals,
                        name=legend_name,
                        marker_color=bar_color,
                        offsetgroup=i,
                        width=bar_width,
                        marker_line_width=0
                    ))
                title = f"Grouped Bar with Separators: {y_axis} by {x_axis} and {group_column}"

            elif plot_type == "Side by Side Bar":
                plot_data = filtered_df[[x_axis, group_column, y_axis]].dropna()
                if plot_data.empty:
                    st.warning("No data available for selected combination.")
                else:
                    means_df = getattr(plot_data.groupby([x_axis, group_column])[y_axis], func_map[agg_func])().reset_index()
                    group_vals = plot_data[group_column].unique()
                    x_vals = plot_data[x_axis].unique()
                    bar_width = 0.8 / len(group_vals) if len(group_vals) else 0.8
                    for i, group in enumerate(group_vals):
                        y_vals = []
                        for x in x_vals:
                            val = plot_data[(plot_data[x_axis] == x) & (plot_data[group_column] == group)][y_axis]
                            y_vals.append(getattr(val, func_map[agg_func])() if len(val) > 0 else 0)
                        legend_name = (
                            legend_names_list[i] if legend_names_list and i < len(legend_names_list)
                            else str(group)
                        )
                        color = side_bar_colors[i] if i < len(side_bar_colors) else px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                        fig.add_trace(go.Bar(
                            x=x_vals,
                            y=y_vals,
                            name=legend_name,
                            marker_color=color,
                            offsetgroup=i,
                            width=bar_width,
                            marker_line_width=0
                        ))
                    if show_average and not means_df.empty:
                        avg = means_df[y_axis].mean()
                        fig.add_hline(y=avg, line_dash='dash', line_color='red', annotation_text=f"Avg: {avg:.2f}", annotation_position="top right")
                    title = f"Side by Side Bar: {agg_func} of {y_axis} by {x_axis} and {group_column}"

            elif plot_type == "Simple Line Plot":
                y_data = filtered_df[[x_axis, y_axis]].dropna()
                legend_name = legend_names_list[0] if legend_names_list else y_axis
                fig.add_trace(go.Scatter(
                    x=y_data[x_axis], y=y_data[y_axis],
                    mode='lines+markers',
                    name=legend_name
                ))
                if show_average and not y_data.empty:
                    avg = y_data[y_axis].mean()
                    fig.add_shape(
                        type="line",
                        x0=min(y_data[x_axis]), x1=max(y_data[x_axis]),
                        y0=avg, y1=avg,
                        line=dict(color="red", dash="dash"),
                        xref='x', yref='y'
                    )
                    fig.add_annotation(
                        x=1, y=avg,
                        xref='paper', yref='y',
                        text=f"Avg: {avg:.2f}",
                        showarrow=False,
                        font=dict(color="red", size=12),
                        xanchor='right',
                        yanchor='bottom'
                    )
                title = f"Line Plot: {y_axis} by {x_axis}"

            elif plot_type == "Grouped Line Plot":
                plot_data = filtered_df[[x_axis, group_column, y_axis]].dropna()
                unique_names = plot_data[group_column].unique()
                for idx, group in enumerate(unique_names):
                    df_group = plot_data[plot_data[group_column] == group]
                    legend_name = (
                        legend_names_list[idx] if legend_names_list and idx < len(legend_names_list)
                        else str(group)
                    )
                    fig.add_trace(go.Scatter(
                        x=df_group[x_axis],
                        y=df_group[y_axis],
                        mode='lines+markers',
                        name=legend_name
                    ))
                title = f"Grouped Line Plot: {y_axis} by {x_axis}, grouped by {group_column}"

            elif plot_type == "Scatter Plot":
                color_col = None if scatter_color == "None" else scatter_color
                size_col = None if scatter_size == "None" else scatter_size
                plot_df = df[[scatter_x_axis, scatter_y_axis] + ([color_col] if color_col else []) + ([size_col] if size_col else [])].dropna()
                if color_col and size_col and color_col in categorical_cols and size_col in numeric_cols:
                    unique_names = plot_df[color_col].unique()
                    for idx, group in enumerate(unique_names):
                        group_df = plot_df[plot_df[color_col] == group]
                        legend_name = (
                            legend_names_list[idx] if legend_names_list and idx < len(legend_names_list)
                            else str(group)
                        )
                        fig.add_trace(go.Scatter(
                            x=group_df[scatter_x_axis],
                            y=group_df[scatter_y_axis],
                            mode='markers',
                            marker=dict(
                                size=group_df[size_col]*10/np.max(group_df[size_col]) if np.max(group_df[size_col]) > 0 else 10,
                                opacity=0.7,
                                line=dict(width=1, color='DarkSlateGrey')
                            ),
                            name=legend_name,
                            text=group_df.index
                        ))
                elif color_col and color_col in categorical_cols:
                    unique_names = plot_df[color_col].unique()
                    for idx, group in enumerate(unique_names):
                        group_df = plot_df[plot_df[color_col] == group]
                        legend_name = (
                            legend_names_list[idx] if legend_names_list and idx < len(legend_names_list)
                            else str(group)
                        )
                        fig.add_trace(go.Scatter(
                            x=group_df[scatter_x_axis],
                            y=group_df[scatter_y_axis],
                            mode='markers',
                            marker=dict(size=10, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')),
                            name=legend_name,
                            text=group_df.index
                        ))
                else:
                    legend_name = legend_names_list[0] if legend_names_list else f"{scatter_y_axis} vs {scatter_x_axis}"
                    fig.add_trace(go.Scatter(
                        x=plot_df[scatter_x_axis],
                        y=plot_df[scatter_y_axis],
                        mode='markers',
                        marker=dict(
                            size=plot_df[size_col]*10/np.max(plot_df[size_col]) if size_col and np.max(plot_df[size_col]) > 0 else 10,
                            color='rgba(30,60,114,0.6)',
                            opacity=0.7,
                            line=dict(width=1, color='DarkSlateGrey')
                        ),
                        name=legend_name,
                        text=plot_df.index
                    ))
                if show_average and not plot_df.empty:
                    avg = plot_df[scatter_y_axis].mean()
                    fig.add_shape(
                        type="line",
                        x0=min(plot_df[scatter_x_axis]), x1=max(plot_df[scatter_x_axis]),
                        y0=avg, y1=avg,
                        line=dict(color="red", dash="dash"),
                        xref='x', yref='y'
                    )
                    fig.add_annotation(
                        x=1, y=avg,
                        xref='paper', yref='y',
                        text=f"Avg: {avg:.2f}",
                        showarrow=False,
                        font=dict(color="red", size=12),
                        xanchor='right',
                        yanchor='bottom'
                    )
                title = f"Scatter Plot: {scatter_y_axis} vs {scatter_x_axis}"

            final_title = custom_title if 'custom_title' in locals() and custom_title else title
            final_x_label = custom_x_label if 'custom_x_label' in locals() and custom_x_label else (
                scatter_x_axis if plot_type == "Scatter Plot" else x_axis
            )
            final_y_label = custom_y_label if 'custom_y_label' in locals() and custom_y_label else (
                scatter_y_axis if plot_type == "Scatter Plot" else y_axis
            )

            if plot_type == "Grouped Bar":
                fig.update_layout(width=plot_width, height=plot_height)
            elif plot_type == "Stacked Bar":
                fig.update_layout(width=plot_width, height=plot_height)

            fig.update_layout(
                title=final_title,
                xaxis_title=final_x_label,
                yaxis_title=final_y_label,
                yaxis=dict(showgrid=False, tickfont=dict(size=yaxis_font_size)),
                xaxis=dict(tickfont=dict(size=xaxis_font_size)),
                bargap=0.05,
                bargroupgap=0.0,
                margin=dict(t=100, b=100)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.charts["Visualization Chart"] = fig

            colA, colB = st.columns(2)
            with colA:
                html_bytes = pio.to_html(fig, full_html=False).encode("utf-8")
                st.download_button(
                    label="Download Interactive HTML",
                    data=html_bytes,
                    file_name=f"plot.html",
                    mime="text/html"
                )
            with colB:
                if "Visualization Chart" in st.session_state.charts:
                    current_fig = st.session_state.charts["Visualization Chart"]
                    if current_fig is not None:
                        img_bytes = save_plotly_as_png(current_fig)
                        if img_bytes is not None:
                            st.download_button(
                                label="Download Static PNG",
                                data=img_bytes,
                                file_name="plot.png",
                                mime="image/png"
                            )
