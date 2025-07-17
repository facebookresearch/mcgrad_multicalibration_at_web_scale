import numpy as np
import numpy.typing as npt
import pandas as pd

from .methods import MCBoost


class CASMCBoostAlgorithm:
    """
    Our own MCBoost implementation.
    """

    def __init__(self, params) -> None:
        """
        Initialize Multicalibration Predictor.
        MCBoost variants:
        - jinetal uses only subgroups as categorical features;
        - nogroups uses only the features and not the groups;
        - alltogether uses both subgroups and features.
        """
        self.mcboost_variant = params["mcboost_variant"] or "jinetal"
        self.weight_column_name = params["weight_column_name"] or None
        self.categorical_feature_column_names = (
            params["categorical_feature_column_names"] or None
        )
        self.numerical_feature_column_names = (
            params["numerical_feature_column_names"] or None
        )
        self.auto_infer_column_types = params.get("auto_infer_column_types", True)
        self.categorical_threshold = params.get("categorical_threshold", 10)

        self.mcboost = MCBoost(
            encode_categorical_variables=params["encode_categorical_variables"] or True,
            monotone_t=params["monotone_t"] or None,
            num_rounds=params["num_rounds"] or 100,
            lightgbm_params=params["lightgbm_params"] or None,
            early_stopping=params["early_stopping"] or None,
            patience=params["patience"] or None,
            early_stopping_score_func=params["early_stopping_score_func"] or None,
            early_stopping_minimize_score=params["early_stopping_minimize_score"]
            or None,
            early_stopping_timeout=params["early_stopping_timeout"] or 8 * 60 * 60,
            save_training_performance=params["save_training_performance"] or False,
            monitored_metrics_during_training=params[
                "monitored_metrics_during_training"
            ]
            or None,
        )

    def fit(self, confs, labels, subgroups, df=None) -> None:
        if self.mcboost_variant in ["nogroups", "alltogether"] and df is None:
            raise ValueError(
                "df has to be passed if variant is nogroups or alltogether"
            )

        df_train, categorical_features = self.preprocess_data(
            confs, subgroups, df, labels
        )

        self.categorical_feature_column_names = (
            categorical_features if len(categorical_features) > 0 else None
        )
        self.numerical_feature_column_names = (
            self.numerical_feature_column_names
            if self.mcboost_variant in ["nogroups", "alltogether"]
            else None
        )

        self.mcboost.fit(
            df_train=df_train,
            prediction_column_name="precali_scores",
            label_column_name="label",
            weight_column_name=self.weight_column_name,
            categorical_feature_column_names=self.categorical_feature_column_names,
            numerical_feature_column_names=self.numerical_feature_column_names,
        )

    def batch_predict(self, f_xs, groups, df=None) -> npt.NDArray:
        if self.mcboost_variant in ["nogroups", "alltogether"] and df is None:
            raise ValueError(
                "df has to be passed if variant is nogroups or alltogether"
            )

        df_test, _ = self.preprocess_data(f_xs, groups, df)

        return self.mcboost.predict(
            df=df_test,
            prediction_column_name="precali_scores",
            categorical_feature_column_names=self.categorical_feature_column_names,
            numerical_feature_column_names=self.numerical_feature_column_names,
            return_all_rounds=False,
        )

    def preprocess_data(self, scores, subgroups=None, df=None, labels=None) -> tuple:
        """
        Preprocess data by creating a dataframe with scores and adding features based on variant.

        Args:
            scores: Array of prediction scores
            subgroups: List of subgroups (for jinetal and alltogether variants)
            df: Original dataframe with additional features (for nogroups and alltogether variants)
            labels: Array of labels (only needed for training)

        Returns:
            tuple: (processed_dataframe, categorical_features_list)
        """
        df_processed = pd.DataFrame(data=scores, columns=["precali_scores"])

        if labels is not None:
            df_processed["label"] = labels

        if self.weight_column_name and df is not None:
            df_processed[self.weight_column_name] = df[self.weight_column_name]

        categorical_features = []

        if self.mcboost_variant == "jinetal":
            categorical_features = self._add_segment_features(df_processed, subgroups)

        elif self.mcboost_variant == "nogroups":
            if df is None:
                raise ValueError("df has to be passed if variant is nogroups")
            categorical_features = self._add_original_features(df_processed, df)

        elif self.mcboost_variant == "alltogether":
            if df is None:
                raise ValueError("df has to be passed if variant is alltogether")
            categorical_features = self._add_segment_features(df_processed, subgroups)
            categorical_features.extend(self._add_original_features(df_processed, df))

        return df_processed, categorical_features

    def _add_segment_features(self, df_processed, subgroups) -> list:
        categorical_features = []
        if subgroups:
            for i, segment in enumerate(subgroups):
                segment_col = f"segment_{i+1}"
                df_processed[segment_col] = [
                    1 if idx in segment else 0 for idx in range(len(df_processed))
                ]
                categorical_features.append(segment_col)
        return categorical_features

    def _infer_column_types(self, df) -> tuple:
        """
        Automatically infer which columns are categorical and which are numerical.

        Args:
            df: DataFrame to analyze

        Returns:
            tuple: (categorical_columns, numerical_columns)
        """
        categorical_columns = []
        numerical_columns = []

        # Skip these columns that are added by our preprocessing
        skip_columns = ["precali_scores", "label"]
        if self.weight_column_name:
            skip_columns.append(self.weight_column_name)

        # Add segment columns to skip list
        segment_columns = [
            col
            for col in df.columns
            if isinstance(col, str) and col.startswith("segment_")
        ]
        skip_columns.extend(segment_columns)

        for col in df.columns:
            if col in skip_columns:
                continue

            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                # If numeric, check if it's likely categorical (few unique values)
                unique_values = df[col].nunique()
                if unique_values <= self.categorical_threshold:
                    categorical_columns.append(col)
                else:
                    numerical_columns.append(col)
            else:
                # Non-numeric columns are treated as categorical
                categorical_columns.append(col)

        print(f"Inferred categorical columns: {categorical_columns}")
        print(f"Inferred numerical columns: {numerical_columns}")

        return categorical_columns, numerical_columns

    def _add_original_features(self, df_processed, df) -> list:
        categorical_features = []

        # Count total available features (excluding special columns)
        skip_columns = ["precali_scores", "label"]
        if self.weight_column_name:
            skip_columns.append(self.weight_column_name)
        segment_columns = [
            col
            for col in df.columns
            if isinstance(col, str) and col.startswith("segment_")
        ]
        skip_columns.extend(segment_columns)

        total_available_features = len(
            [col for col in df.columns if col not in skip_columns]
        )

        # If auto-inference is enabled and no column types are specified, infer them
        if self.auto_infer_column_types and (
            self.categorical_feature_column_names is None
            or self.numerical_feature_column_names is None
        ):
            inferred_cat_cols, inferred_num_cols = self._infer_column_types(df)

            if self.categorical_feature_column_names is None:
                self.categorical_feature_column_names = inferred_cat_cols

            if self.numerical_feature_column_names is None:
                self.numerical_feature_column_names = inferred_num_cols

        used_cat_columns = []
        if self.categorical_feature_column_names:
            existing_cat_columns = [
                col
                for col in self.categorical_feature_column_names
                if col in df.columns
            ]
            if existing_cat_columns:
                df_processed[existing_cat_columns] = df[existing_cat_columns]
                categorical_features.extend(existing_cat_columns)
                used_cat_columns = existing_cat_columns
            else:
                print(
                    f"Warning: None of the categorical feature columns {self.categorical_feature_column_names} exist in the DataFrame. Available columns: {df.columns.tolist()}"
                )

        used_num_columns = []
        if self.numerical_feature_column_names:
            existing_num_columns = [
                col for col in self.numerical_feature_column_names if col in df.columns
            ]
            if existing_num_columns:
                df_processed[existing_num_columns] = df[existing_num_columns]
                used_num_columns = existing_num_columns
            else:
                print(
                    f"Warning: None of the numerical feature columns {self.numerical_feature_column_names} exist in the DataFrame. Available columns: {df.columns.tolist()}"
                )

        # Print feature usage summary
        total_used_features = len(used_cat_columns) + len(used_num_columns)
        print(f"\nFeature usage in {self.mcboost_variant} variant:")
        print(f"  Total available features: {total_available_features}")
        print(
            f"  Used features: {total_used_features} ({total_used_features/total_available_features:.1%} of available)"
        )
        print(f"    - Categorical: {len(used_cat_columns)}")
        print(f"    - Numerical: {len(used_num_columns)}")

        return categorical_features
