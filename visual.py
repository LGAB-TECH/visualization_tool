import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

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

with st.sidebar:
    st.markdown("## File Selection Mode")
    file_mode = st.radio(
        "Choose analysis mode:",
        ("Single CSV Analysis", "Multiple CSV Merge & Analysis"),
        help="Select if you want to analyze a single CSV or merge multiple CSVs before analysis."
    )

df = None

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

    # Always set df from session_state if exists
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
    "📊 Bar Plot"
]
if "tab_radio" not in st.session_state:
    st.session_state["tab_radio"] = tab_labels[0]
selected_tab = st.radio("Navigation", tab_labels, horizontal=True, key="tab_radio")

def goto_bar_plot_tab():
    st.session_state["tab_radio"] = tab_labels[2]
    st.session_state["show_bar_plot"] = True
    st.rerun()  # Use st.rerun() instead of st.experimental_rerun()

if df is not None:
    # Clean and keep track of columns that had missing values filled
    df_clean, cleaned_columns, allna_cols = clean_and_fill_all_but_allna(df)
    df = df_clean  # Use only cleaned data globally

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
        st.markdown("<h3 style='color:#1e3c72;'>📊 Custom Bar Plot</h3>", unsafe_allow_html=True)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if categorical_cols and numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                plot_type = st.selectbox(
                    "📊 Plot Type:",
                    ["Grouped Bar", "Stacked Bar", "Grouped Bar with Separators"],
                    help="""
                    Grouped Bar: Standard bar plot
                    Stacked Bar: Bars stacked on top of each other
                    Grouped Bar with Separators: Bars with vertical separators between groups
                    """
                )
                x_axis = st.selectbox("🧭 Select X-axis (categorical):", categorical_cols)
                x_axis_values = df[x_axis].unique().tolist()
                selected_x_values = st.multiselect(
                    f"Select {x_axis} values to display:",
                    options=x_axis_values,
                    default=x_axis_values
                )
                filtered_df = df[df[x_axis].isin(selected_x_values)]
                if plot_type != "Grouped Bar with Separators":
                    agg_func = st.selectbox(
                        "📐 Aggregation Function:",
                        ["Sum", "Mean", "Median", "Count", "Standard Deviation", "Minimum", "Maximum"],
                        help="""
                        Sum: Total of values
                        Mean: Average of values
                        Median: Middle value
                        Count: Number of records
                        Standard Deviation: Measure of variation
                        Minimum: Smallest value
                        Maximum: Largest value
                        """
                    )
            with col2:
                y_axis = st.selectbox("📏 Select Y-axis (numeric):", numeric_cols)
                if plot_type == "Grouped Bar with Separators":
                    group_column = st.selectbox(
                        "🏷️ Select Group Column:",
                        [col for col in categorical_cols if col != x_axis],
                        help="Select a categorical column to group by"
                    )
                    bar_color = st.color_picker("🎨 Select Bar Color:", "#FF0000")
                else:
                    show_average = st.checkbox("Show Average Line", value=True)
                if plot_type == "Stacked Bar":
                    stack_column = st.selectbox(
                        "🔢 Select Stacking Column:",
                        [col for col in categorical_cols if col != x_axis],
                        help="Select a categorical column to stack by"
                    )

            if st.button("📈 Generate Plot", on_click=goto_bar_plot_tab):
                st.session_state["show_bar_plot"] = True
            # Plot only if explicitly requested
            if st.session_state.get("show_bar_plot", False):
                fig = go.Figure()
                if plot_type == "Grouped Bar":
                    group_df = filtered_df[[x_axis, y_axis]].groupby(x_axis)
                    if agg_func == "Sum":
                        y_data = group_df.sum().reset_index()
                    elif agg_func == "Mean":
                        y_data = group_df.mean().reset_index()
                    elif agg_func == "Median":
                        y_data = group_df.median().reset_index()
                    elif agg_func == "Count":
                        y_data = group_df.count().reset_index()
                    elif agg_func == "Standard Deviation":
                        y_data = group_df.std().reset_index()
                    elif agg_func == "Minimum":
                        y_data = group_df.min().reset_index()
                    elif agg_func == "Maximum":
                        y_data = group_df.max().reset_index()
                    else:
                        y_data = group_df.sum().reset_index()
                    fig.add_trace(go.Bar(
                        x=y_data[x_axis],
                        y=y_data[y_axis],
                        marker_color="#1e3c72"
                    ))
                    if show_average:
                        avg = y_data[y_axis].mean()
                        fig.add_shape(
                            type="line",
                            x0=min(y_data[x_axis]), x1=max(y_data[x_axis]),
                            y0=avg, y1=avg,
                            line=dict(color="red", dash="dash"),
                        )
                        fig.add_annotation(
                            x=0.95, y=avg, xref="paper", yref="y", text=f"Avg: {avg:.2f}",
                            showarrow=False, font=dict(color="red"), align="right"
                        )
                    fig.update_layout(
                        barmode='group',
                        xaxis_title=x_axis,
                        yaxis_title=y_axis,
                        title=f"{agg_func} of {y_axis} by {x_axis}"
                    )
                elif plot_type == "Stacked Bar":
                    group_df = filtered_df[[x_axis, stack_column, y_axis]]
                    if agg_func == "Sum":
                        plot_df = group_df.groupby([x_axis, stack_column])[y_axis].sum().reset_index()
                    elif agg_func == "Mean":
                        plot_df = group_df.groupby([x_axis, stack_column])[y_axis].mean().reset_index()
                    elif agg_func == "Median":
                        plot_df = group_df.groupby([x_axis, stack_column])[y_axis].median().reset_index()
                    elif agg_func == "Count":
                        plot_df = group_df.groupby([x_axis, stack_column])[y_axis].count().reset_index()
                    elif agg_func == "Standard Deviation":
                        plot_df = group_df.groupby([x_axis, stack_column])[y_axis].std().reset_index()
                    elif agg_func == "Minimum":
                        plot_df = group_df.groupby([x_axis, stack_column])[y_axis].min().reset_index()
                    elif agg_func == "Maximum":
                        plot_df = group_df.groupby([x_axis, stack_column])[y_axis].max().reset_index()
                    else:
                        plot_df = group_df.groupby([x_axis, stack_column])[y_axis].sum().reset_index()
                    for s in plot_df[stack_column].unique():
                        df_s = plot_df[plot_df[stack_column] == s]
                        fig.add_trace(go.Bar(
                            x=df_s[x_axis],
                            y=df_s[y_axis],
                            name=str(s)
                        ))
                    if show_average:
                        avg = plot_df[y_axis].mean()
                        fig.add_shape(
                            type="line",
                            x0=min(plot_df[x_axis]), x1=max(plot_df[x_axis]),
                            y0=avg, y1=avg,
                            line=dict(color="red", dash="dash"),
                        )
                        fig.add_annotation(
                            x=0.95, y=avg, xref="paper", yref="y", text=f"Avg: {avg:.2f}",
                            showarrow=False, font=dict(color="red"), align="right"
                        )
                    fig.update_layout(
                        barmode='stack',
                        xaxis_title=x_axis,
                        yaxis_title=y_axis,
                        title=f"{agg_func} of {y_axis} by {x_axis} and {stack_column}"
                    )
                else:  # Grouped Bar with Separators
                    plot_data = filtered_df[[x_axis, group_column, y_axis]]
                    group_vals = plot_data[group_column].unique()
                    x_vals = plot_data[x_axis].unique()
                    bar_width = 0.8 / len(group_vals)
                    for i, group in enumerate(group_vals):
                        y_vals = []
                        for x in x_vals:
                            val = plot_data[(plot_data[x_axis] == x) & (plot_data[group_column] == group)][y_axis]
                            y_vals.append(val.mean() if len(val) > 0 else 0)
                        fig.add_trace(go.Bar(
                            x=x_vals,
                            y=y_vals,
                            name=str(group),
                            marker_color=bar_color,
                            offsetgroup=i,
                            width=bar_width
                        ))
                    for idx in range(1, len(x_vals)):
                        fig.add_shape(
                            type="line",
                            x0=idx-0.5, x1=idx-0.5,
                            y0=0, y1=max([max(trace.y) if len(trace.y)>0 else 0 for trace in fig.data]),
                            line=dict(color="gray", dash="dash")
                        )
                    fig.update_layout(
                        barmode='group',
                        xaxis_title=x_axis,
                        yaxis_title=y_axis,
                        title=f"Grouped Bar with Separators: {y_axis} by {x_axis} and {group_column}"
                    )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("<h4 style='color:#2a5298;margin-top:20px;'>📊 Summary Statistics</h4>", unsafe_allow_html=True)
                if plot_type == "Grouped Bar":
                    stats_df = filtered_df[[x_axis, y_axis]][y_axis].describe().round(2)
                    st.dataframe(
                        pd.DataFrame(stats_df).T.style.background_gradient(cmap='Blues'),
                        use_container_width=True
                    )
                elif plot_type == "Stacked Bar":
                    stats_df = filtered_df[[x_axis, stack_column, y_axis]].groupby(stack_column)[y_axis].describe().round(2)
                    st.dataframe(
                        stats_df.style.background_gradient(cmap='Blues'),
                        use_container_width=True
                    )
                else:
                    stats_df = plot_data.groupby(group_column)[y_axis].describe().round(2)
                    st.dataframe(
                        stats_df.style.background_gradient(cmap='Blues'),
                        use_container_width=True
                    )
        else:
            st.info("🧾 Need both categorical and numeric columns for plotting!")

else:
    st.markdown("👆 <span style='color:#2a5298;font-size:17px;'>Upload a dataset to get started.</span>", unsafe_allow_html=True)
