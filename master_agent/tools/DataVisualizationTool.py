import matplotlib
matplotlib.use("Agg")  # ✅ Force non-GUI backend before importing pyplot

import json
import base64
import pandas as pd
import seaborn as sns
from io import BytesIO
import matplotlib.pyplot as plt
from pydantic import BaseModel
from master_agent.tools.BaseTool import BaseTool

class DataVisualizationTool(BaseTool):
    """Tool for analyzing datasets and generating visualizations."""

    description = "Analyzes datasets (CSV, Excel, JSON) and generates visualizations."

    class InputSchema(BaseModel):
        file_path: str  # Path to the dataset file
        analysis_type: str  # "summary", "describe", "histogram", "line_chart", "bar_chart", "scatter_plot"
        column_x: str = None  # Optional: Column for x-axis
        column_y: str = None  # Optional: Column for y-axis

    def run(self, arguments: dict):
        """Performs data analysis and generates plots."""
        args = self.validate_args(arguments)
        file_path = args.file_path
        analysis_type = args.analysis_type.lower().strip()  # Normalize case & trim spaces
        column_x = args.column_x
        column_y = args.column_y

        # Supported analysis types
        valid_analysis_types = {"summary", "describe", "histogram", "line_chart", "bar_chart", "scatter_plot"}

        try:
            # Debugging Step: Print received arguments
            print(f"Debug - Received Args: {args}")

            # Load data
            df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
            print(f"Data Loaded Successfully. Columns: {df.columns.tolist()}")

            # Validate analysis type
            if analysis_type not in valid_analysis_types:
                return json.dumps({"error": f"Invalid analysis type: {analysis_type}. Supported types: {list(valid_analysis_types)}"})

            # Summary or description analysis
            if analysis_type == "summary":
                return json.dumps({"columns": list(df.columns), "missing_values": df.isnull().sum().to_dict()})

            elif analysis_type == "describe":
                return df.describe().to_json()

            # Validate columns for visualization
            if analysis_type in {"histogram", "line_chart", "bar_chart", "scatter_plot"}:
                if column_x not in df.columns or (column_y and column_y not in df.columns):
                    return json.dumps({"error": f"Invalid column names: {column_x}, {column_y}. Ensure they exist in the dataset."})

                print(f"Generating Plot: {analysis_type} - X: {column_x}, Y: {column_y}")

                plt.clf()  # Clear previous figures
                plt.figure(figsize=(6, 4), dpi=100)
                sns.set(style="whitegrid")

                # Create the requested plot
                if analysis_type == "histogram":
                    sns.histplot(df[column_x], bins=20, kde=True)
                    plt.xlabel(column_x)
                    plt.ylabel("Frequency")

                elif analysis_type == "line_chart":
                    sns.lineplot(data=df, x=column_x, y=column_y)
                    plt.xlabel(column_x)
                    plt.ylabel(column_y)

                elif analysis_type == "bar_chart":
                    sns.barplot(data=df, x=column_x, y=column_y)
                    plt.xlabel(column_x)
                    plt.ylabel(column_y)

                elif analysis_type == "scatter_plot":
                    sns.scatterplot(data=df, x=column_x, y=column_y)
                    plt.xlabel(column_x)
                    plt.ylabel(column_y)

                plt.title(f"{analysis_type.capitalize()} of {column_x} vs {column_y}")

                # Convert the plot to a base64-encoded string
                img_buffer = BytesIO()
                plt.savefig(img_buffer, format="png", bbox_inches='tight')  # Prevent cropping
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.read()).decode()

                plt.close()  # Close the figure to free memory
                print("Plot successfully generated.")

                return json.dumps({"image": img_base64})

        except Exception as e:
            print(f"Error: {str(e)}")
            return json.dumps({"error": str(e)})
