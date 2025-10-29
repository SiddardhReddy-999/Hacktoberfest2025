import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import plotly.express as px
from datetime import datetime, timedelta

@st.cache_data
def generate_sample_data(start_time=None, periods=200, freq_seconds=1):
    if start_time is None:
        start_time = datetime.utcnow() - timedelta(seconds=periods*freq_seconds)
    times = [start_time + timedelta(seconds=i*freq_seconds) for i in range(periods)]
    np.random.seed(42)
    base = np.cumsum(np.random.randn(periods))
    metric_a = base + np.random.normal(scale=2.0, size=periods)
    metric_b = np.sin(np.linspace(0, 6.28, periods)) * 10 + np.random.normal(scale=1.5, size=periods)
    metric_c = np.abs(np.random.randn(periods) * 5)
    df = pd.DataFrame({
        'timestamp': times,
        'metric_a': metric_a,
        'metric_b': metric_b,
        'metric_c': metric_c,
    })
    return df

def parse_uploaded_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        try:
            return pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            return None

def to_csv_bytes(df: pd.DataFrame):
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer

st.set_page_config(page_title="Data Dashboard with Live Charts", layout="wide")
st.title("📊 Data Dashboard — Live Charts")

with st.sidebar:
    st.header("Data Source")
    use_sample = st.radio("Choose data source:", ("Sample generated data", "Upload CSV"))
    uploaded = None
    if use_sample == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx', 'xls'])
    st.markdown("---")
    st.header("Display Options")
    chart_type = st.selectbox("Chart type", ['Line', 'Area', 'Bar', 'Scatter'])
    x_col = st.text_input("X column (leave blank to auto-detect)")
    y_cols = st.text_input("Y columns (comma-separated, or leave blank for all numeric)")
    rolling_window = st.number_input("Moving average window (points, 0 = off)", min_value=0, value=0, step=1)
    st.markdown("---")
    st.header("Live Mode")
    live_mode = st.checkbox("Enable live updating (auto-refresh)", value=False)
    refresh_seconds = st.number_input("Refresh interval (seconds)", min_value=1, value=3, step=1)
    max_points = st.number_input("Max points to show (for performance)", min_value=50, value=200, step=10)
    st.markdown("---")
    st.caption("Made with ❤️ — upload small files for best performance")

if use_sample == "Sample generated data":
    if 'sample_start' not in st.session_state:
        st.session_state.sample_start = datetime.utcnow() - timedelta(seconds=200)
    if 'sample_df' not in st.session_state:
        st.session_state.sample_df = generate_sample_data(st.session_state.sample_start, periods=200, freq_seconds=1)
    if live_mode:
        new_row_count = 1
        last_ts = st.session_state.sample_df['timestamp'].max()
        new_start = last_ts + timedelta(seconds=1)
        new_df = generate_sample_data(start_time=new_start, periods=new_row_count, freq_seconds=1)
        st.session_state.sample_df = pd.concat([st.session_state.sample_df, new_df], ignore_index=True)
    df = st.session_state.sample_df.copy()
else:
    if uploaded is None:
        st.info("Upload a CSV/Excel file to get started, or switch to sample data.")
        st.stop()
    df = parse_uploaded_csv(uploaded)
    if df is None:
        st.stop()

if x_col.strip() == "":
    for candidate in ['timestamp', 'time', 'datetime', 'date']:
        if candidate in df.columns:
            x_col = candidate
            break

if x_col in df.columns:
    try:
        df[x_col] = pd.to_datetime(df[x_col])
    except Exception:
        pass
else:
    df = df.reset_index().rename(columns={'index': 'index'})
    x_col = 'index'

if y_cols.strip() == "":
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if x_col in numeric_cols:
        numeric_cols.remove(x_col)
    y_columns = numeric_cols
else:
    y_columns = [c.strip() for c in y_cols.split(',') if c.strip() in df.columns]

if len(y_columns) == 0:
    st.warning("No numeric y-columns found. Please select columns in the Y columns field or upload data with numeric columns.")

if x_col in df.columns and df[x_col].dtype == 'datetime64[ns]':
    df = df.sort_values(by=x_col)

if len(df) > max_points:
    df = df.tail(max_points)

if rolling_window and rolling_window > 0 and y_columns:
    df_smooth = df.copy()
    for col in y_columns:
        if col in df_smooth.columns and pd.api.types.is_numeric_dtype(df_smooth[col]):
            df_smooth[col] = df_smooth[col].rolling(window=rolling_window, min_periods=1).mean()
    df_display = df_smooth
else:
    df_display = df

col1, col2 = st.columns((3,1))
with col1:
    st.subheader("Live Chart")
    if len(y_columns) == 0:
        st.write("No columns to plot.")
    else:
        if x_col in df_display.columns:
            df_plot = df_display[[x_col] + y_columns].melt(id_vars=x_col, value_vars=y_columns, var_name='series', value_name='value')
            if chart_type == 'Line':
                fig = px.line(df_plot, x=x_col, y='value', color='series', markers=False)
            elif chart_type == 'Area':
                fig = px.area(df_plot, x=x_col, y='value', color='series', line_group='series')
            elif chart_type == 'Bar':
                fig = px.bar(df_plot, x=x_col, y='value', color='series', barmode='group')
            else:
                fig = px.scatter(df_plot, x=x_col, y='value', color='series')
            fig.update_layout(height=500, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("X column not present in data")
    st.markdown("---")
    st.subheader("Summary Statistics")
    st.write(df_display[y_columns].describe())

with col2:
    st.subheader("Controls & Filters")
    if x_col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[x_col]):
            min_dt = df[x_col].min()
            max_dt = df[x_col].max()
            selected_range = st.slider("Time range", min_value=min_dt, max_value=max_dt, value=(min_dt, max_dt))
            df_display = df_display[(df_display[x_col] >= selected_range[0]) & (df_display[x_col] <= selected_range[1])]
        else:
            st.write(f"X column type: {df[x_col].dtype}")
    st.write("\n")
    st.markdown("### Export & View")
    st.download_button("Download CSV of displayed data", data=to_csv_bytes(df_display), file_name='dashboard_data.csv')
    st.write("\n")
    show_table = st.checkbox("Show raw table", value=False)
    if show_table:
        st.dataframe(df_display)

st.markdown("---")
metrics = df_display[y_columns].iloc[-1] if len(df_display)>0 and y_columns else None
if metrics is not None:
    cols = st.columns(len(y_columns))
    for i, colname in enumerate(y_columns):
        try:
            cols[i].metric(label=colname, value=round(metrics[colname], 3))
        except Exception:
            cols[i].metric(label=colname, value=str(metrics[colname]))

if live_mode:
    st.experimental_rerun()

st.info("Tip: Use 'Sample generated data' to see live updates. Upload a CSV to visualize your own dataset.")
