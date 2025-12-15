import json
import math
import os

import numpy as np
import pandas as pd


class MetricsQuadrants:
    def __init__(self, data: pd.DataFrame, metrics: list[str]):
        self.data = data
        self.metrics = metrics
        self.types = self._nest_by_columns(data, ['graph_type'])
        self.types_quadrants = self._nest_by_columns(data, ['graph_type', 'quadrant'])
        self.types_quadrants_correctness = self._nest_by_columns(data, ['graph_type', 'quadrant', 'is_correct'])


    @staticmethod
    def _load_data(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path, sep=',')


    def _save_metrics(self, data: dict, file_path: str) -> None:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f)


    def _separate_by_column(self, data: pd.DataFrame, column: str) -> dict[str, pd.DataFrame]:
        unique_values = data[column].unique()
        return {value: data[data[column] == value] for value in unique_values}


    # Nested dictionary of dataframes by columns
    def _nest_by_columns(self, df: pd.DataFrame, columns: list[str]) -> dict:
        # Recursive function based on level that upload it until reaching the leaves 
        def build(level_df: pd.DataFrame, level: int) -> dict:
            column = columns[level]
            groups = self._separate_by_column(level_df, column)
            if level == len(columns) - 1:
                return groups
            return {value: build(sub_df, level + 1) for value, sub_df in groups.items()}
        return build(df, 0)


    @staticmethod
    def _weighted_mean(values: list[float], weights: list[float]) -> float:
        if not values:
            return 0.0
        values_arr = np.array(values, dtype=float)
        weights_arr = np.array(weights, dtype=float)
        total_weight = weights_arr.sum()
        if total_weight == 0:
            return 0.0
        return float(np.dot(values_arr, weights_arr) / total_weight)

    @staticmethod
    def _weighted_std(values: list[float], weights: list[float]) -> float:
        if not values:
            return 0.0
        values_arr = np.array(values, dtype=float)
        weights_arr = np.array(weights, dtype=float)
        total_weight = weights_arr.sum()
        if total_weight == 0:
            return 0.0
        mean = np.dot(values_arr, weights_arr) / total_weight
        variance = np.dot(weights_arr, (values_arr - mean) ** 2) / total_weight
        return float(math.sqrt(variance))

    @staticmethod
    def _safe_mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _safe_std(values: list[float]) -> float:
        return float(np.std(values)) if values else 0.0

    def quadrant_base_metrics(self, output_path: str | None = None) -> dict:
        quadrant_metrics = {}
        for graph_type, quadrants in self.types_quadrants.items():
            quadrant_metrics[graph_type] = {}
            for quadrant, quadrant_df in quadrants.items():
                experiment_total = int(quadrant_df['experiment_total'].iloc[0])
                correctness_map = self.types_quadrants_correctness[graph_type][quadrant]
                correct_df = correctness_map.get(True)
                incorrect_df = correctness_map.get(False)

                total_correct = int(correct_df['count'].iloc[0]) if correct_df is not None else 0
                total_incorrect = int(incorrect_df['count'].iloc[0]) if incorrect_df is not None else 0
                total = total_correct + total_incorrect

                quadrant_metrics[graph_type][quadrant] = {}

                # Percentage of samples falling into this quadrant w.r.t. experiment_total
                quadrant_metrics[graph_type][quadrant]['Total Percentage'] = (
                    (total / experiment_total) * 100 if experiment_total else 0
                )

                if total > 0:
                    correct_percentage = (total_correct / total) * 100
                    incorrect_percentage = (total_incorrect / total) * 100
                else:
                    correct_percentage = 0
                    incorrect_percentage = 0

                quadrant_metrics[graph_type][quadrant]['Correct Percentage'] = correct_percentage
                quadrant_metrics[graph_type][quadrant]['Incorrect Percentage'] = incorrect_percentage

                difference = correct_percentage - incorrect_percentage
                quadrant_metrics[graph_type][quadrant]['Correctness difference'] = difference
                quadrant_metrics[graph_type][quadrant]['Correctness absolute difference'] = abs(difference)

        if output_path:
            self._save_metrics(quadrant_metrics, output_path)
        return quadrant_metrics

    def quadrant_composed_metrics(self, output_path: str | None = None, base_metrics: dict | None = None) -> dict:
        base_metrics = base_metrics or self.quadrant_base_metrics()
        composed = {}
        for graph_type, quadrant_data in base_metrics.items():
            weights = []
            correct_percentages = []
            incorrect_percentages = []
            deltas = []
            abs_deltas = []
            for metrics in quadrant_data.values():
                weight = float(metrics.get('Total Percentage', 0.0)) / 100.0
                weights.append(weight)
                correct_val = float(metrics.get('Correct Percentage', 0.0))
                incorrect_val = float(metrics.get('Incorrect Percentage', 0.0))
                delta_val = float(metrics.get('Correctness difference', 0.0))
                correct_percentages.append(correct_val)
                incorrect_percentages.append(incorrect_val)
                deltas.append(delta_val)
                abs_deltas.append(abs(delta_val))

            mean_separation_unweighted = self._safe_mean(abs_deltas)
            mean_separation_weighted = self._weighted_mean(abs_deltas, weights)
            delta_std = self._safe_std(deltas)

            sd_correct_unweighted = self._safe_std(correct_percentages)
            sd_correct_weighted = self._weighted_std(correct_percentages, weights)
            sd_incorrect_unweighted = self._safe_std(incorrect_percentages)
            sd_incorrect_weighted = self._weighted_std(incorrect_percentages, weights)

            combined_sep_unweighted = math.sqrt(
                sd_correct_unweighted ** 2 + sd_incorrect_unweighted ** 2
            )
            combined_sep_weighted = math.sqrt(
                sd_correct_weighted ** 2 + sd_incorrect_weighted ** 2
            )

            composed[graph_type] = {
                'Mean Separation Unweighted': mean_separation_unweighted,
                'Mean Separation Weighted': mean_separation_weighted,
                'Separation Std Dev': delta_std,
                'SD Correct Unweighted': sd_correct_unweighted,
                'SD Correct Weighted': sd_correct_weighted,
                'SD Incorrect Unweighted': sd_incorrect_unweighted,
                'SD Incorrect Weighted': sd_incorrect_weighted,
                'Combined Separability Unweighted': combined_sep_unweighted,
                'Combined Separability Weighted': combined_sep_weighted,
            }

        if output_path:
            self._save_metrics(composed, output_path)
        return composed



if __name__ == '__main__':
    datasets = ['setfit_ag_news', 'stanfordnlp_sst2']
    for dataset in datasets:
        csv_path = f'/app/outputs/analytics/consistency/plots/{dataset}/vizB_quadrant_summary_{dataset}.csv'
        data = MetricsQuadrants._load_data(csv_path)
        metrics_quadrants = MetricsQuadrants(data, ['experimental_total'])
        base_output = f'outputs/analytics/consistency/metrics/quadrant_base_metrics_{dataset}.json'
        composed_output = f'outputs/analytics/consistency/metrics/quadrant_composed_metrics_{dataset}.json'
        quadrant_metrics = metrics_quadrants.quadrant_base_metrics(base_output)
        quadrant_composed_metrics = metrics_quadrants.quadrant_composed_metrics(composed_output, quadrant_metrics)
        print(f'{dataset}: {quadrant_composed_metrics}')

    print("Finished!")
