import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.data_loader import load_dataset, DATASETS_INFO
from classifiers.models import get_classifiers
from utils.metrics import calculate_metrics

from algorithms.registry import OPTIMIZERS
from feature_selection.optimizer_wrapper import (
    OptimizerFeatureSelectionWrapper
)


# =========================================================
# Parameter Settings
# =========================================================

ALGORITHMS_TO_RUN = [
    "SFO",
    # Classical algorithms
    "PSO",
    "GWO",
    "DE",
    "GA",
    "ACOR",
    # Well-established comparison algorithms
    "WOA",
    "HHO",
    # Recent algorithms
    "HBA",
    "AGTO",
    "MGO",
    "SeaHO",
    "CoatiOA",
    "SCSO",
]

N_RUNS = 10
POP_SIZE = 10
MAX_ITER = 200

OUTER_TEST_SIZE = 0.30
INNER_VALIDATION_SIZE = 0.20

LOWER_BOUND = -6.0
UPPER_BOUND = 6.0

DATASETS_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "datasets"
)

RESULTS_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "results"
)

RAW_RESULTS_FILE = os.path.join(
    RESULTS_DIRECTORY,
    "optimization_comparison_raw_results.csv"
)

SUMMARY_RESULTS_FILE = os.path.join(
    RESULTS_DIRECTORY,
    "optimization_comparison_summary_results.csv"
)

ACCURACY_TABLE_FILE = os.path.join(
    RESULTS_DIRECTORY,
    "optimization_comparison_accuracy_table.csv"
)

DEFECTIVE_METRICS_FILE = os.path.join(
    RESULTS_DIRECTORY,
    "optimization_comparison_defective_metrics.csv"
)


RESUME_EXISTING_RESULTS = True


# =========================================================
# create results folder
# =========================================================

def create_result_directories():
    directories = [
        RESULTS_DIRECTORY,
        os.path.join(RESULTS_DIRECTORY, "convergence"),
        os.path.join(RESULTS_DIRECTORY, "selected_features"),
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# =========================================================

def can_use_stratification(y):
    classes, counts = np.unique(
        y,
        return_counts=True
    )

    return (
        len(classes) >= 2
        and counts.min() >= 2
    )


def safe_train_test_split(
    X,
    y,
    test_size,
    random_state
):
    stratify_values = (
        y if can_use_stratification(y)
        else None
    )

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values
    )


# =========================================================
# save convergence cureve
# =========================================================

def save_convergence(
    algorithm_name,
    dataset_name,
    classifier_name,
    run_number,
    fitness_curve,
    accuracy_curve
):
    if len(fitness_curve) != len(accuracy_curve):
        raise ValueError(
            "Fitness and accuracy curves must have equal lengths."
        )

    curve_df = pd.DataFrame({
        "Algorithm": algorithm_name,
        "Dataset": dataset_name,
        "Classifier": classifier_name,
        "Run": run_number,
        "Iteration": np.arange(
            1,
            len(fitness_curve) + 1
        ),
        "Best_Fitness": fitness_curve,
        "Best_Validation_Accuracy": accuracy_curve
    })

    filename = (
        f"{algorithm_name}_"
        f"{dataset_name}_"
        f"{classifier_name}_"
        f"Run{run_number}.csv"
    )

    file_path = os.path.join(
        RESULTS_DIRECTORY,
        "convergence",
        filename
    )

    curve_df.to_csv(
        file_path,
        index=False
    )


# =========================================================
# save selected features for each run
# =========================================================

def save_selected_features(
    algorithm_name,
    dataset_name,
    classifier_name,
    run_number,
    selected_indices,
    feature_names
):
    filename = (
        f"{algorithm_name}_"
        f"{dataset_name}_"
        f"{classifier_name}_"
        f"Run{run_number}.txt"
    )

    file_path = os.path.join(
        RESULTS_DIRECTORY,
        "selected_features",
        filename
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"Algorithm: {algorithm_name}\n"
        )

        file.write(
            f"Dataset: {dataset_name}\n"
        )

        file.write(
            f"Classifier: {classifier_name}\n"
        )

        file.write(
            f"Run: {run_number}\n"
        )

        file.write(
            f"Number of selected features: "
            f"{len(selected_indices)}\n\n"
        )

        for index in selected_indices:
            file.write(
                f"{int(index)}: "
                f"{feature_names[int(index)]}\n"
            )


# =========================================================
# تحميل النتائج السابقة للاستكمال
# =========================================================

def load_existing_results():
    if (
        RESUME_EXISTING_RESULTS
        and os.path.exists(RAW_RESULTS_FILE)
    ):
        existing_df = pd.read_csv(
            RAW_RESULTS_FILE
        )

        print(
            f"Existing raw results loaded: "
            f"{len(existing_df)} rows"
        )

        return existing_df

    return pd.DataFrame()


def build_completed_experiment_keys(existing_df):
    completed_keys = set()

    required_columns = {
        "Algorithm",
        "Dataset",
        "Classifier",
        "Run"
    }

    if (
        existing_df.empty
        or not required_columns.issubset(
            existing_df.columns
        )
    ):
        return completed_keys

    for _, row in existing_df.iterrows():
        completed_keys.add((
            str(row["Algorithm"]),
            str(row["Dataset"]),
            str(row["Classifier"]),
            int(row["Run"])
        ))

    return completed_keys


# =========================================================
# حفظ النتائج الخام تدريجياً
# =========================================================

def save_progress(results_records):
    progress_df = pd.DataFrame(
        results_records
    )

    progress_df.to_csv(
        RAW_RESULTS_FILE,
        index=False
    )


# =========================================================
# إنشاء الجداول النهائية
# =========================================================

def generate_summary_tables(results_df):

    summary_columns = [
        "Accuracy",

        "Precision_Defective",
        "Recall_Defective",
        "F1_Defective",

        "Precision_Weighted",
        "Recall_Weighted",
        "F1_Weighted",

        "RMSE",

        "Selected_Features",

        "Best_Validation_Fitness",
        "Best_Validation_Accuracy",

        "Time_Seconds"
    ]

    available_summary_columns = [
        column
        for column in summary_columns
        if column in results_df.columns
    ]

    summary_df = (
        results_df
        .groupby([
            "Algorithm",
            "Dataset",
            "Classifier"
        ])[available_summary_columns]
        .agg([
            "mean",
            "std",
            "min",
            "max"
        ])
    )

    summary_df.columns = [
        "_".join(column).strip()
        for column in summary_df.columns.values
    ]

    summary_df = summary_df.reset_index()

    summary_df.to_csv(
        SUMMARY_RESULTS_FILE,
        index=False
    )

    # -----------------------------------------------------
    # جدول الدقة المشابه لجدول ورقة FSBOA
    # -----------------------------------------------------

    accuracy_table = (
        results_df
        .groupby([
            "Algorithm",
            "Dataset",
            "Classifier"
        ])["Accuracy"]
        .mean()
        .reset_index()
        .pivot_table(
            index=[
                "Algorithm",
                "Dataset"
            ],
            columns="Classifier",
            values="Accuracy"
        )
        .reset_index()
    )

    accuracy_table.columns.name = None

    accuracy_table.to_csv(
        ACCURACY_TABLE_FILE,
        index=False
    )

    # -----------------------------------------------------
    # جدول مقاييس الفئة المعيبة
    # -----------------------------------------------------

    defective_metrics = (
        results_df
        .groupby([
            "Algorithm",
            "Dataset",
            "Classifier"
        ])
        .agg(
            Accuracy_Mean=(
                "Accuracy",
                "mean"
            ),
            Precision_Defective_Mean=(
                "Precision_Defective",
                "mean"
            ),
            Recall_Defective_Mean=(
                "Recall_Defective",
                "mean"
            ),
            F1_Defective_Mean=(
                "F1_Defective",
                "mean"
            ),
            RMSE_Mean=(
                "RMSE",
                "mean"
            ),
            Selected_Features_Mean=(
                "Selected_Features",
                "mean"
            ),
            Time_Seconds_Mean=(
                "Time_Seconds",
                "mean"
            )
        )
        .reset_index()
    )

    defective_metrics.to_csv(
        DEFECTIVE_METRICS_FILE,
        index=False
    )

    return (
        summary_df,
        accuracy_table,
        defective_metrics
    )


# =========================================================
# البرنامج الرئيس
# =========================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning
    )

    create_result_directories()

    existing_df = load_existing_results()

    if existing_df.empty:
        results_records = []
    else:
        results_records = (
            existing_df
            .to_dict("records")
        )

    completed_keys = build_completed_experiment_keys(
        existing_df
    )

    classifiers = get_classifiers(
        random_state=42
    )

    total_expected_runs = (
        len(ALGORITHMS_TO_RUN)
        * len(DATASETS_INFO)
        * len(classifiers)
        * N_RUNS
    )

    print("\n" + "=" * 100)
    print("Optimization Algorithms Comparison")
    print("=" * 100)

    print(
        f"Algorithms: {ALGORITHMS_TO_RUN}"
    )

    print(
        f"Datasets: {list(DATASETS_INFO.keys())}"
    )

    print(
        f"Classifiers: {list(classifiers.keys())}"
    )

    print(
        f"Population size: {POP_SIZE}"
    )

    print(
        f"Iterations: {MAX_ITER}"
    )

    print(
        f"Independent runs: {N_RUNS}"
    )

    print(
        f"Expected experiments: {total_expected_runs}"
    )

    print(
        f"Already completed: {len(completed_keys)}"
    )

    # =====================================================
    # حلقة الخوارزميات
    # =====================================================

    for algorithm_name in ALGORITHMS_TO_RUN:

        if algorithm_name not in OPTIMIZERS:
            raise KeyError(
                f"Algorithm '{algorithm_name}' "
                f"is not defined in OPTIMIZERS registry."
            )

        algorithm_config = OPTIMIZERS[
            algorithm_name
        ]

        optimizer_class = algorithm_config[
            "class"
        ]

        optimizer_parameters = algorithm_config.get(
            "parameters",
            {}
        )

        implementation_source = algorithm_config.get(
            "source",
            "Not specified"
        )

        print("\n" + "#" * 100)
        print(
            f"Algorithm: {algorithm_name}"
        )
        print(
            f"Source: {implementation_source}"
        )
        print("#" * 100)

        # =================================================
        # حلقة قواعد البيانات
        # =================================================

        for dataset_name in DATASETS_INFO.keys():

            print("\n" + "=" * 90)
            print(
                f"Algorithm: {algorithm_name} | "
                f"Dataset: {dataset_name}"
            )
            print("=" * 90)

            (
                X_dataframe,
                y_series,
                _,
                target_column
            ) = load_dataset(
                dataset_name,
                datasets_dir=DATASETS_DIRECTORY
            )

            feature_names = list(
                X_dataframe.columns
            )

            X_full = X_dataframe.to_numpy(
                dtype=float
            )

            y_full = y_series.to_numpy(
                dtype=int
            )

            class_values, class_counts = np.unique(
                y_full,
                return_counts=True
            )

            class_distribution = dict(
                zip(
                    class_values.tolist(),
                    class_counts.tolist()
                )
            )

            print(
                f"Samples: {X_full.shape[0]}"
            )

            print(
                f"Features: {X_full.shape[1]}"
            )

            print(
                f"Target: {target_column}"
            )

            print(
                f"Class distribution: "
                f"{class_distribution}"
            )

            # =============================================
            # حلقة المصنفات
            # =============================================

            for classifier_name, classifier in classifiers.items():

                print(
                    f"\nClassifier: {classifier_name}"
                )

                # =========================================
                # حلقة التشغيلات المستقلة
                # =========================================

                for run_number in range(
                    1,
                    N_RUNS + 1
                ):

                    experiment_key = (
                        algorithm_name,
                        dataset_name,
                        classifier_name,
                        run_number
                    )

                    if experiment_key in completed_keys:
                        print(
                            f"Run {run_number}/{N_RUNS} "
                            f"already completed — skipped."
                        )
                        continue

                    print(
                        f"Run {run_number}/{N_RUNS}"
                    )

                    try:
                        # ---------------------------------
                        # Outer split:
                        # الاختبار النهائي لا تراه الخوارزمية
                        # ---------------------------------

                        (
                            X_outer_train,
                            X_independent_test,
                            y_outer_train,
                            y_independent_test
                        ) = safe_train_test_split(
                            X=X_full,
                            y=y_full,
                            test_size=OUTER_TEST_SIZE,
                            random_state=run_number
                        )

                        # ---------------------------------
                        # Inner split:
                        # يستخدم في اختيار الميزات
                        # ---------------------------------

                        (
                            X_inner_train,
                            X_validation,
                            y_inner_train,
                            y_validation
                        ) = safe_train_test_split(
                            X=X_outer_train,
                            y=y_outer_train,
                            test_size=INNER_VALIDATION_SIZE,
                            random_state=run_number
                        )

                        # ---------------------------------
                        # Scaling لمرحلة Feature Selection
                        # ---------------------------------

                        inner_scaler = StandardScaler()

                        X_inner_train_scaled = (
                            inner_scaler.fit_transform(
                                X_inner_train
                            )
                        )

                        X_validation_scaled = (
                            inner_scaler.transform(
                                X_validation
                            )
                        )

                        start_time = time.perf_counter()

                        # ---------------------------------
                        # Wrapper العام
                        # ---------------------------------

                        feature_selector = (
                            OptimizerFeatureSelectionWrapper(
                                optimizer_class=optimizer_class,
                                classifier=classifier,
                                optimizer_parameters=(
                                    optimizer_parameters
                                ),
                                population_size=POP_SIZE,
                                max_iter=MAX_ITER,
                                lb=LOWER_BOUND,
                                ub=UPPER_BOUND,
                                random_state=run_number
                            )
                        )

                        feature_selector.fit(
                            X_inner_train=(
                                X_inner_train_scaled
                            ),
                            X_validation=(
                                X_validation_scaled
                            ),
                            y_inner_train=(
                                y_inner_train
                            ),
                            y_validation=(
                                y_validation
                            )
                        )

                        selected_indices = np.asarray(
                            feature_selector.selected_features,
                            dtype=int
                        )

                        if selected_indices.size == 0:
                            raise RuntimeError(
                                "The optimizer returned "
                                "an empty feature subset."
                            )

                        # ---------------------------------
                        # التقييم النهائي المستقل
                        # ---------------------------------

                        final_scaler = StandardScaler()

                        X_outer_train_scaled = (
                            final_scaler.fit_transform(
                                X_outer_train
                            )
                        )

                        X_independent_test_scaled = (
                            final_scaler.transform(
                                X_independent_test
                            )
                        )

                        X_outer_train_selected = (
                            X_outer_train_scaled[
                                :,
                                selected_indices
                            ]
                        )

                        X_independent_test_selected = (
                            X_independent_test_scaled[
                                :,
                                selected_indices
                            ]
                        )

                        metrics = calculate_metrics(
                            classifier=classifier,
                            X_train_selected=(
                                X_outer_train_selected
                            ),
                            X_test_selected=(
                                X_independent_test_selected
                            ),
                            y_train=y_outer_train,
                            y_test=y_independent_test
                        )

                        elapsed_time = (
                            time.perf_counter()
                            - start_time
                        )

                        # ---------------------------------
                        # حفظ منحنى التقارب
                        # ---------------------------------

                        save_convergence(
                            algorithm_name=algorithm_name,
                            dataset_name=dataset_name,
                            classifier_name=classifier_name,
                            run_number=run_number,
                            fitness_curve=(
                                feature_selector
                                .convergence_curve
                            ),
                            accuracy_curve=(
                                feature_selector
                                .accuracy_curve
                            )
                        )

                        # ---------------------------------
                        # حفظ الميزات المختارة
                        # ---------------------------------

                        save_selected_features(
                            algorithm_name=algorithm_name,
                            dataset_name=dataset_name,
                            classifier_name=classifier_name,
                            run_number=run_number,
                            selected_indices=selected_indices,
                            feature_names=feature_names
                        )

                        result_row = {
                            "Algorithm": algorithm_name,

                            "Implementation_Source":
                                implementation_source,

                            "Dataset": dataset_name,

                            "Classifier":
                                classifier_name,

                            "Run": run_number,

                            "Random_Seed":
                                run_number,

                            "Population_Size":
                                POP_SIZE,

                            "Max_Iterations":
                                MAX_ITER,

                            "Lower_Bound":
                                LOWER_BOUND,

                            "Upper_Bound":
                                UPPER_BOUND,

                            "Total_Samples":
                                X_full.shape[0],

                            "Original_Features":
                                X_full.shape[1],

                            "Outer_Train_Samples":
                                len(y_outer_train),

                            "Independent_Test_Samples":
                                len(y_independent_test),

                            "Inner_Train_Samples":
                                len(y_inner_train),

                            "Validation_Samples":
                                len(y_validation),

                            "Best_Validation_Fitness":
                                feature_selector.best_fitness,

                            "Best_Validation_Accuracy":
                                (
                                    1.0
                                    - feature_selector.best_fitness
                                ) * 100.0,

                            "Selected_Features":
                                len(selected_indices),

                            "Selected_Feature_Indices":
                                selected_indices.tolist(),

                            "Selected_Feature_Names": [
                                feature_names[index]
                                for index in selected_indices
                            ],

                            "Time_Seconds":
                                elapsed_time,

                            "Status":
                                "Completed",

                            "Error_Message":
                                ""
                        }

                        result_row.update(
                            metrics
                        )

                        results_records.append(
                            result_row
                        )

                        completed_keys.add(
                            experiment_key
                        )

                        save_progress(
                            results_records
                        )

                        print(
                            f"Completed | "
                            f"Accuracy: "
                            f"{metrics['Accuracy']:.4f}% | "
                            f"Features: "
                            f"{len(selected_indices)} | "
                            f"Time: "
                            f"{elapsed_time:.2f} s"
                        )

                    except Exception as error:

                        print(
                            f"ERROR in "
                            f"{algorithm_name} - "
                            f"{dataset_name} - "
                            f"{classifier_name} - "
                            f"Run {run_number}"
                        )

                        print(
                            str(error)
                        )

                        error_row = {
                            "Algorithm": algorithm_name,

                            "Implementation_Source":
                                implementation_source,

                            "Dataset":
                                dataset_name,

                            "Classifier":
                                classifier_name,

                            "Run":
                                run_number,

                            "Random_Seed":
                                run_number,

                            "Population_Size":
                                POP_SIZE,

                            "Max_Iterations":
                                MAX_ITER,

                            "Status":
                                "Failed",

                            "Error_Message":
                                str(error)
                        }

                        results_records.append(
                            error_row
                        )

                        save_progress(
                            results_records
                        )

    # =====================================================
    # إنشاء النتائج النهائية
    # =====================================================

    results_df = pd.DataFrame(
        results_records
    )

    results_df.to_csv(
        RAW_RESULTS_FILE,
        index=False
    )

    completed_results_df = results_df[
        results_df["Status"] == "Completed"
    ].copy()

    if completed_results_df.empty:
        raise RuntimeError(
            "No experiment was completed successfully."
        )

    (
        summary_df,
        accuracy_table,
        defective_metrics
    ) = generate_summary_tables(
        completed_results_df
    )

    print("\n" + "=" * 100)
    print("All requested experiments are finished.")
    print("=" * 100)

    print(
        f"\nRaw results:\n"
        f"{RAW_RESULTS_FILE}"
    )

    print(
        f"\nSummary results:\n"
        f"{SUMMARY_RESULTS_FILE}"
    )

    print(
        f"\nAccuracy comparison table:\n"
        f"{ACCURACY_TABLE_FILE}"
    )

    print(
        f"\nDefective-class metrics:\n"
        f"{DEFECTIVE_METRICS_FILE}"
    )

    print(
        f"\nSuccessful experiments: "
        f"{len(completed_results_df)}"
    )

    failed_results = results_df[
        results_df["Status"] == "Failed"
    ]

    print(
        f"Failed experiments: "
        f"{len(failed_results)}"
    )


if __name__ == "__main__":
    main()