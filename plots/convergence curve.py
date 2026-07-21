import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




RESULTS_DIRECTORY = "results"

CONVERGENCE_DIRECTORY = os.path.join(
    RESULTS_DIRECTORY,
    "convergence"
)

OUTPUT_DIRECTORY = os.path.join(
    RESULTS_DIRECTORY,
    "plots",
    "fitness_convergence_by_classifier"
)




ALGORITHM_ORDER = [
    "SFO",
    "PSO",
    "GWO",
    "DE",
    "GA",
    "ACOR",
    "WOA",
    "HHO",
    "HBA",
    "AGTO",
    "MGO",
    "SeaHO",
    "CoatiOA",
    "SCSO",
]


DISPLAY_NAMES = {
    "SFO": "FSSFO",
    "PSO": "FSPSO",
    "GWO": "FSGWO",
    "DE": "FSDE",
    "GA": "FSGA",
    "ACOR": "FSACOR",
    "WOA": "FSWOA",
    "HHO": "FSHHO",
    "HBA": "FSHBA",
    "AGTO": "FSAGTO",
    "MGO": "FSMGO",
    "SeaHO": "FSSeaHO",
    "CoatiOA": "FSCoatiOA",
    "SCSO": "FSSCSO",
}


DATASET_ORDER = [
    "CM1", "JM1", "KC1", "KC3",
    "KC4", "MC1", "MC2", "PC1",
    "PC2", "PC3", "PC4", "PC5",
]

CLASSIFIER_ORDER = [
    "NB", "KNN", "DT", "LDA",
]


USE_SHARED_LEGEND = True

N_ROWS = 3
N_COLUMNS = 4
FIGURE_SIZE = (20, 13)


REQUIRED_COLUMNS = {
    "Dataset",
    "Classifier",
    "Run",
    "Iteration",
    "Best_Fitness",
}




def safe_filename(text):
   
    text = str(text).strip()
    text = re.sub(r"[^\w\-]+", "_", text)
    return text.strip("_")


def standardize_algorithm_name(name):

    name = str(name).strip()

    for algorithm in ALGORITHM_ORDER:
        if name.lower() == algorithm.lower():
            return algorithm

        if name.lower() == DISPLAY_NAMES[algorithm].lower():
            return algorithm

    return name


def infer_algorithm_from_filename(file_path):

    file_name = os.path.splitext(
        os.path.basename(file_path)
    )[0]

    normalized_file_name = re.sub(
        r"[^a-zA-Z0-9]",
        "",
        file_name
    ).lower()

    candidates = sorted(
        ALGORITHM_ORDER,
        key=len,
        reverse=True
    )

    for algorithm in candidates:
        possible_names = [
            algorithm,
            DISPLAY_NAMES[algorithm],
        ]

        for possible_name in possible_names:
            normalized_name = re.sub(
                r"[^a-zA-Z0-9]",
                "",
                possible_name
            ).lower()

            if normalized_name in normalized_file_name:
                return algorithm

    return None




def load_convergence_files():

    pattern = os.path.join(
        CONVERGENCE_DIRECTORY,
        "*.csv"
    )

    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            "No convergence CSV files were found in:\n"
            f"{CONVERGENCE_DIRECTORY}"
        )

    frames = []
    skipped_files = []

    for file_path in files:
        try:
            df = pd.read_csv(file_path)
        except Exception as error:
            skipped_files.append(
                (file_path, f"Read error: {error}")
            )
            continue

        missing_columns = REQUIRED_COLUMNS.difference(
            df.columns
        )

        if missing_columns:
            skipped_files.append(
                (
                    file_path,
                    f"Missing columns: {sorted(missing_columns)}"
                )
            )
            continue

        if "Algorithm" in df.columns:
            df["Algorithm"] = (
                df["Algorithm"]
                .astype(str)
                .map(standardize_algorithm_name)
            )

        else:
            algorithm = infer_algorithm_from_filename(
                file_path
            )

            if algorithm is None:
                skipped_files.append(
                    (
                        file_path,
                        "Algorithm column is absent and the "
                        "algorithm name cannot be inferred "
                        "from the file name."
                    )
                )
                continue

            df["Algorithm"] = algorithm

        df["Source_File"] = os.path.basename(
            file_path
        )

        frames.append(df)

    if skipped_files:
        print("\nSkipped files:")

        for file_path, reason in skipped_files:
            print(f"- {file_path}")
            print(f"  Reason: {reason}")

    if not frames:
        raise RuntimeError(
            "No compatible convergence CSV files were found."
        )

    convergence_df = pd.concat(
        frames,
        ignore_index=True
    )

    convergence_df["Dataset"] = (
        convergence_df["Dataset"]
        .astype(str)
        .str.strip()
    )

    convergence_df["Classifier"] = (
        convergence_df["Classifier"]
        .astype(str)
        .str.strip()
    )

    convergence_df["Algorithm"] = (
        convergence_df["Algorithm"]
        .astype(str)
        .str.strip()
        .map(standardize_algorithm_name)
    )

    convergence_df["Run"] = pd.to_numeric(
        convergence_df["Run"],
        errors="raise"
    )

    convergence_df["Iteration"] = pd.to_numeric(
        convergence_df["Iteration"],
        errors="raise"
    ).astype(int)

    convergence_df["Best_Fitness"] = pd.to_numeric(
        convergence_df["Best_Fitness"],
        errors="raise"
    )

    convergence_df = convergence_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    convergence_df = convergence_df.dropna(
        subset=[
            "Dataset",
            "Classifier",
            "Algorithm",
            "Run",
            "Iteration",
            "Best_Fitness",
        ]
    )

    if convergence_df.empty:
        raise RuntimeError(
            "No valid convergence records remain after cleaning."
        )

    return convergence_df




def calculate_mean_fitness_curves(convergence_df):

    mean_curves = (
        convergence_df
        .groupby(
            [
                "Dataset",
                "Classifier",
                "Algorithm",
                "Iteration",
            ],
            as_index=False
        )
        .agg(
            Mean_Best_Fitness=(
                "Best_Fitness",
                "mean"
            ),
            Number_of_Runs=(
                "Run",
                "nunique"
            )
        )
    )

    return mean_curves


def save_mean_curves(mean_curves):
    """
    حفظ القيم المستخدمة في الرسوم.
    """
    os.makedirs(
        RESULTS_DIRECTORY,
        exist_ok=True
    )

    output_path = os.path.join(
        RESULTS_DIRECTORY,
        "all_algorithms_mean_fitness_convergence.csv"
    )

    mean_curves.to_csv(
        output_path,
        index=False
    )

    print(
        "\nMean convergence data saved to:\n"
        f"{output_path}"
    )


def report_available_algorithms(mean_curves):
 
    coverage = (
        mean_curves
        .groupby(
            ["Dataset", "Classifier"]
        )["Algorithm"]
        .agg(
            Number_of_Algorithms="nunique",
            Algorithms=lambda values: ", ".join(
                sorted(set(values))
            )
        )
        .reset_index()
    )

    print("\nAvailable algorithms for each figure:")
    print(coverage.to_string(index=False))

    incomplete = coverage[
        coverage["Number_of_Algorithms"]
        < len(ALGORITHM_ORDER)
    ]

    if not incomplete.empty:
        print(
            "\nWarning: Some Dataset-Classifier combinations "
            "do not contain all 14 algorithms."
        )



def plot_fitness_curves(mean_curves):

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True
    )

    available_classifiers = list(
        mean_curves["Classifier"].dropna().unique()
    )

    classifiers = [
        classifier
        for classifier in CLASSIFIER_ORDER
        if classifier in available_classifiers
    ]

    classifiers.extend(
        sorted(
            set(available_classifiers)
            .difference(classifiers)
        )
    )

    generated_count = 0

    for classifier_name in classifiers:

        classifier_data = mean_curves[
            mean_curves["Classifier"] == classifier_name
        ].copy()

        if classifier_data.empty:
            continue

        available_datasets = list(
            classifier_data["Dataset"].dropna().unique()
        )

        datasets = [
            dataset
            for dataset in DATASET_ORDER
            if dataset in available_datasets
        ]

        datasets.extend(
            sorted(
                set(available_datasets)
                .difference(datasets)
            )
        )

        required_rows = max(
            N_ROWS,
            int(np.ceil(len(datasets) / N_COLUMNS))
        )

        figure, axes = plt.subplots(
            required_rows,
            N_COLUMNS,
            figsize=FIGURE_SIZE,
            squeeze=False
        )

        axes = axes.flatten()

        shared_handles = []
        shared_labels = []

        for subplot_index, dataset_name in enumerate(datasets):

            axis = axes[subplot_index]

            dataset_data = classifier_data[
                classifier_data["Dataset"] == dataset_name
            ]

            available_algorithms = set(
                dataset_data["Algorithm"].unique()
            )

            algorithms_to_plot = [
                algorithm
                for algorithm in ALGORITHM_ORDER
                if algorithm in available_algorithms
            ]

            algorithms_to_plot.extend(
                sorted(
                    available_algorithms.difference(
                        algorithms_to_plot
                    )
                )
            )

            for algorithm in algorithms_to_plot:

                curve = dataset_data[
                    dataset_data["Algorithm"] == algorithm
                ].sort_values("Iteration")

                if curve.empty:
                    continue

                line, = axis.plot(
                    curve["Iteration"],
                    curve["Mean_Best_Fitness"],
                    linewidth=1.25,
                    label=DISPLAY_NAMES.get(
                        algorithm,
                        algorithm
                    )
                )

                display_label = DISPLAY_NAMES.get(
                    algorithm,
                    algorithm
                )

                if display_label not in shared_labels:
                    shared_handles.append(line)
                    shared_labels.append(display_label)

            axis.set_title(
                f"Fitness for {dataset_name} data sets- "
                f"{classifier_name} classifier",
                fontsize=10
            )

            axis.set_xlabel(
                "No of iterations",
                fontsize=9
            )

            axis.set_ylabel(
                "Error Value",
                fontsize=9
            )

            axis.grid(
                True,
                alpha=0.40,
                linewidth=0.7
            )

            axis.set_xlim(left=0)
            axis.tick_params(
                axis="both",
                labelsize=8
            )

            if not USE_SHARED_LEGEND:
                axis.legend(
                    loc="upper right",
                    fontsize=5.5,
                    frameon=True,
                    ncol=1
                )

        for subplot_index in range(
            len(datasets),
            len(axes)
        ):
            axes[subplot_index].set_visible(False)

        figure.suptitle(
            f"Convergence curves for {classifier_name} classifier",
            fontsize=16,
            y=0.995
        )

        if USE_SHARED_LEGEND and shared_handles:
            figure.legend(
                shared_handles,
                shared_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.005),
                ncol=7,
                fontsize=8,
                frameon=True
            )

            figure.tight_layout(
                rect=[0.02, 0.075, 0.98, 0.965]
            )

        else:
            figure.tight_layout(
                rect=[0.02, 0.02, 0.98, 0.965]
            )

        safe_classifier = safe_filename(
            classifier_name
        )

        file_name = (
            f"Convergence_All_Datasets_"
            f"{safe_classifier}"
        )

        png_path = os.path.join(
            OUTPUT_DIRECTORY,
            f"{file_name}.png"
        )

        pdf_path = os.path.join(
            OUTPUT_DIRECTORY,
            f"{file_name}.pdf"
        )

        figure.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight"
        )

        figure.savefig(
            pdf_path,
            bbox_inches="tight"
        )

        plt.close(figure)

        print(f"Generated: {png_path}")
        generated_count += 1

    return generated_count


# =========================================================
# Main
# =========================================================

def main():
    convergence_df = load_convergence_files()

    mean_curves = calculate_mean_fitness_curves(
        convergence_df
    )

    save_mean_curves(
        mean_curves
    )

    report_available_algorithms(
        mean_curves
    )

    generated_count = plot_fitness_curves(
        mean_curves
    )

    print(
        "\nComposite convergence figures generated: "
        f"{generated_count}"
    )

    print(
        "\nFigures saved in:\n"
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
