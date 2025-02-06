import json
import tempfile
import csv
import streamlit as st
import pandas as pd
from phi.model.openai import OpenAIChat
from phi.agent.duckdb import DuckDbAgent
from phi.tools.pandas import PandasTools
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional
import re


# Function to create visualizations
def create_visualization(df: pd.DataFrame, viz_type: str,
                         x_column: Optional[str] = None,
                         y_columns: Optional[List[str]] = None,
                         color_column: Optional[str] = None,
                         title: str = "") -> go.Figure:
    """
    Create various types of visualizations based on the specified parameters.
    """
    try:
        if viz_type == "bar":
            fig = px.bar(df, x=x_column, y=y_columns[0] if y_columns else None,
                         color=color_column, title=title)
        elif viz_type == "line":
            fig = px.line(df, x=x_column, y=y_columns if y_columns else None,
                          color=color_column, title=title)
        elif viz_type == "scatter":
            fig = px.scatter(df, x=x_column, y=y_columns[0] if y_columns else None,
                             color=color_column, title=title)
        elif viz_type == "pie":
            fig = px.pie(df, values=y_columns[0] if y_columns else None,
                         names=x_column, title=title)
        elif viz_type == "histogram":
            fig = px.histogram(df, x=x_column, color=color_column, title=title)
        elif viz_type == "box":
            fig = px.box(df, x=x_column, y=y_columns[0] if y_columns else None,
                         color=color_column, title=title)
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")

        fig.update_layout(
            title_x=0.5,
            margin=dict(t=50, l=0, r=0, b=0),
            height=500
        )
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None


def suggest_visualization(df: pd.DataFrame, columns: List[str], query: str) -> dict:
    """
    Suggest appropriate visualization based on the data and query.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    time_keywords = ['trend', 'over time', 'temporal', 'evolution', 'change']
    comparison_keywords = ['compare', 'comparison', 'difference', 'versus', 'vs']
    distribution_keywords = ['distribution', 'spread', 'range', 'histogram']

    viz_params = {
        'viz_type': 'bar',
        'x_column': None,
        'y_columns': None,
        'color_column': None,
        'title': ''
    }

    query = query.lower()

    if any(keyword in query for keyword in time_keywords) and datetime_cols:
        viz_params['viz_type'] = 'line'
        viz_params['x_column'] = datetime_cols[0]
        if numeric_cols:
            viz_params['y_columns'] = [numeric_cols[0]]

    elif any(keyword in query for keyword in distribution_keywords):
        if numeric_cols:
            viz_params['viz_type'] = 'histogram'
            viz_params['x_column'] = numeric_cols[0]

    elif any(keyword in query for keyword in comparison_keywords):
        if categorical_cols and numeric_cols:
            viz_params['viz_type'] = 'bar'
            viz_params['x_column'] = categorical_cols[0]
            viz_params['y_columns'] = [numeric_cols[0]]

    if not viz_params['x_column']:
        if datetime_cols:
            viz_params['x_column'] = datetime_cols[0]
        elif categorical_cols:
            viz_params['x_column'] = categorical_cols[0]
        elif numeric_cols:
            viz_params['x_column'] = numeric_cols[0]

    if not viz_params['y_columns'] and numeric_cols:
        viz_params['y_columns'] = [numeric_cols[0]]

    viz_params['title'] = query.capitalize()

    return viz_params


# Function to preprocess and save the uploaded file
def preprocess_and_save(file):
    try:
        # Read the uploaded file into a DataFrame
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None

        # Ensure string columns are properly quoted
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)

        # Parse dates and numeric columns
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass

        # Create a temporary file to save the preprocessed data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)

        return temp_path, df.columns.tolist(), df

    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None


# Streamlit app
st.title("📊 Data Analyst Agent")

# Sidebar for API keys
with st.sidebar:
    st.header("API Keys")
    openai_key = st.text_input("Enter your OpenAI API key:", type="password")
    if openai_key:
        st.session_state.openai_key = openai_key
        st.success("API key saved!")
    else:
        st.warning("Please enter your OpenAI API key to proceed.")

# File upload widget
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None and "openai_key" in st.session_state:
    # Preprocess and save the uploaded file
    temp_path, columns, df = preprocess_and_save(uploaded_file)

    if temp_path and columns and df is not None:
        # Display the uploaded data as a table
        st.write("Uploaded Data:")
        st.dataframe(df)

        # Display the columns of the uploaded data
        st.write("Uploaded columns:", columns)

        # Configure the semantic model
        semantic_model = {
            "tables": [
                {
                    "name": "uploaded_data",
                    "description": "Contains the uploaded dataset.",
                    "path": temp_path,
                }
            ]
        }

        # Initialize the DuckDbAgent
        duckdb_agent = DuckDbAgent(
            model=OpenAIChat(model="gpt-4", api_key=st.session_state.openai_key),
            semantic_model=json.dumps(semantic_model),
            tools=[PandasTools()],
            markdown=True,
            add_history_to_messages=False,
            followups=False,
            read_tool_call_history=False,
            system_prompt="You are an expert data analyst. Generate SQL queries to solve the user's query. Return only the SQL query, enclosed in ```sql ``` and give the final answer.",
        )

        # Initialize code storage in session state
        if "generated_code" not in st.session_state:
            st.session_state.generated_code = None

        # Main query input widget
        user_query = st.text_area("Ask a query about the data:")

        if st.button("Submit Query"):
            if user_query.strip() == "":
                st.warning("Please enter a query.")
            else:
                try:
                    # Show loading spinner while processing
                    with st.spinner('Processing your query...'):
                        # Get the response from DuckDbAgent
                        response1 = duckdb_agent.run(user_query)

                        # Extract the content from the RunResponse object
                        if hasattr(response1, 'content'):
                            response_content = response1.content
                        else:
                            response_content = str(response1)
                        response = duckdb_agent.print_response(
                            user_query,
                            stream=True,
                        )

                    # Display the response in Streamlit
                    st.markdown(response_content)

                    # Visualization section
                    st.markdown("---")
                    st.subheader("📈 Visualization")

                    # Create columns for visualization controls
                    col1, col2, col3 = st.columns([2, 2, 1])

                    with col1:
                        viz_type = st.selectbox(
                            "Select visualization type:",
                            ["bar", "line", "scatter", "pie", "histogram", "box"],
                            index=0
                        )

                    # Get suggested visualization parameters
                    suggested_viz = suggest_visualization(df, columns, user_query)

                    with col2:
                        x_column = st.selectbox(
                            "Select X-axis column:",
                            columns,
                            index=columns.index(suggested_viz['x_column']) if suggested_viz[
                                                                                  'x_column'] in columns else 0
                        )

                    with col3:
                        color_column = st.selectbox(
                            "Select color column (optional):",
                            ["None"] + columns,
                            index=0
                        )

                    # For y-axis selection (multiple possible for line charts)
                    y_columns = st.multiselect(
                        "Select Y-axis column(s):",
                        columns,
                        default=suggested_viz['y_columns'] if suggested_viz['y_columns'] else []
                    )

                    # Create visualization
                    if y_columns or viz_type in ['histogram', 'pie']:
                        fig = create_visualization(
                            df,
                            viz_type=viz_type,
                            x_column=x_column,
                            y_columns=y_columns,
                            color_column=None if color_column == "None" else color_column,
                            title=f"{viz_type.capitalize()} Chart: {user_query}"
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error generating response from the DuckDbAgent: {e}")
                    st.error("Please try rephrasing your query or check if the data format is correct.")