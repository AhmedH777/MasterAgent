import matplotlib
matplotlib.use("Agg")  # ✅ Force non-GUI backend before importing pyplot

import os
import json
import pandas as pd
import seaborn as sns
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
            # Load data
            df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)

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

                # Save plot in a project 'temp' directory
                project_dir = os.path.dirname(os.path.abspath(__file__))
                temp_dir = os.path.join(project_dir, '..', '..', 'temp')
                os.makedirs(temp_dir, exist_ok=True)  # Create temp folder if it doesn't exist

                file_name = os.path.join(temp_dir, "plot.png")
                plt.savefig(file_name, format="png", bbox_inches='tight')
                plt.close()

                # Return JSON with relative path
                return json.dumps({
                    "message": f"The data has been read successfully, and a {analysis_type} visualizing the relationship between {column_x} and {column_y} is ready.",
                    "image_url": "/temp/plot.png"
                })


        except Exception as e:
            return json.dumps({"error": str(e)})
