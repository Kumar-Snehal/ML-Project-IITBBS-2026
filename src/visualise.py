"""
Generate presentation-ready visualizations for the EPL Match Prediction project.
Reads generated CSVs from reports/ and features/ (with exact fallback data)
and outputs high-resolution PNG charts into reports/.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try locating project root whether script is in ROOT or ROOT/src/
CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR if (CURRENT_DIR / "reports").exists() else CURRENT_DIR.parent
REPORTS_DIR = ROOT / "reports"
FEATURES_DIR = ROOT / "features"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Global styling for clean academic presentation slides
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


def plot_1_class_balance():
    """Plot class distribution across Train, Validation, and Test chronological splits."""
    splits = [
        "Train\n(2000-01 to 2012-13)\nn = 4,940",
        "Validation\n(2013-14 to 2014-15)\nn = 760",
        "Test\n(2015-16 to 2017-18)\nn = 1,140",
    ]
    away = [27.2, 31.3, 29.2]
    draw = [26.1, 22.5, 25.4]
    home = [46.7, 46.2, 45.4]

    x = np.arange(len(splits))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    rects_a = ax.bar(x - width, away, width, label="Away Win (A)", color="#e74c3c")
    rects_d = ax.bar(x, draw, width, label="Draw (D)", color="#95a5a6")
    rects_h = ax.bar(x + width, home, width, label="Home Win (H)", color="#2980b9")

    ax.axhline(
        33.33,
        color="#2c3e50",
        linestyle=":",
        linewidth=1.2,
        label="Uniform Argmax Threshold (33.3%)",
    )
    ax.set_ylabel("Percentage of Matches (%)")
    ax.set_title("Chronological Data Splits & Class Balance (EPL 2000–2018)")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylim(0, 56)
    ax.legend(frameon=True, loc="upper right")

    for rects in [rects_a, rects_d, rects_h]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    fig.tight_layout()
    out = REPORTS_DIR / "plot_1_class_balance.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_2_mutual_information():
    """Plot Mutual Information of all 19 features and highlight selected subsets."""
    signal_file = REPORTS_DIR / "feature_signal.csv"
    if signal_file.exists():
        df = pd.read_csv(signal_file)
    else:
        df = pd.DataFrame(
            [
                ("elo_diff", "strength", 0.0859),
                ("ppg_diff", "season", 0.0614),
                ("gd_pg_diff", "season", 0.0582),
                ("form5_diff", "form", 0.0534),
                ("venue_ppg_diff", "season", 0.0490),
                ("sot5_diff", "form", 0.0484),
                ("ga5_diff", "form", 0.0255),
                ("gf5_diff", "form", 0.0161),
                ("h2h_pts", "h2h", 0.0150),
                ("abs_elo_diff", "strength", 0.0052),
                ("matchweek", "context", 0.0043),
                ("away_promoted", "context", 0.0036),
                ("draw_rate_home", "draw", 0.0025),
                ("draw_rate_away", "draw", 0.0000),
                ("total_goals10", "draw", 0.0000),
                ("rest_home", "context", 0.0000),
                ("h2h_n", "h2h", 0.0000),
                ("rest_away", "context", 0.0000),
                ("home_promoted", "context", 0.0000),
            ],
            columns=["feature", "group", "mutual_information"],
        )

    df = df.sort_values("mutual_information", ascending=True).reset_index(drop=True)

    nb_selected = {"elo_diff", "matchweek", "away_promoted", "draw_rate_home"}
    lr_selected = {
        "elo_diff",
        "form5_diff",
        "venue_ppg_diff",
        "sot5_diff",
        "ga5_diff",
        "gf5_diff",
        "h2h_pts",
        "abs_elo_diff",
        "matchweek",
        "away_promoted",
        "draw_rate_home",
    }

    group_colors = {
        "strength": "#1f77b4",
        "season": "#ff7f0e",
        "form": "#2ca02c",
        "h2h": "#9467bd",
        "context": "#8c564b",
        "draw": "#e377c2",
    }
    colors = [group_colors.get(g, "#7f7f7f") for g in df["group"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        df["feature"],
        df["mutual_information"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel("Mutual Information with Match Outcome (Train Set)")
    ax.set_title("Feature Ranking by Mutual Information & Model Selection Tags")
    ax.set_xlim(0, 0.102)

    for bar, feat, val in zip(bars, df["feature"], df["mutual_information"]):
        tag = ""
        if feat in nb_selected and feat in lr_selected:
            tag = "  [LR + NB]"
        elif feat in lr_selected:
            tag = "  [LR only]"
        elif val > 0:
            tag = "  [Dropped: Collinear |r|>0.85]"
        else:
            tag = "  [Dropped: MI=0]"

        ax.text(
            val + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}{tag}",
            va="center",
            ha="left",
            fontsize=8.5,
        )

    # Legend for feature groups
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=g.capitalize())
        for g, c in group_colors.items()
    ]
    ax.legend(handles=handles, title="Feature Group", loc="lower right", frameon=True)

    fig.tight_layout()
    out = REPORTS_DIR / "plot_2_mutual_information.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_3_hyperparameter_tuning():
    """Plot Validation Log Loss curves for Naive Bayes (bins) and Logistic Regression (C)."""
    nb_bins = [3, 4, 5, 6, 8, 10]
    nb_logloss_sel = [0.9873, 0.9811, 0.9766, 0.9784, 0.9779, 0.9785]
    nb_acc_sel = [0.5566, 0.5553, 0.5526, 0.5474, 0.5553, 0.5500]

    lr_c = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    lr_logloss = [0.9746, 0.9693, 0.9677, 0.9678, 0.9683, 0.9685, 0.9686, 0.9686]
    lr_rps = [0.2025, 0.2006, 0.1998, 0.1996, 0.1996, 0.1996, 0.1996, 0.1996]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left: Naive Bayes Bins
    ax1.plot(
        nb_bins,
        nb_logloss_sel,
        marker="o",
        linewidth=2.2,
        color="#d35400",
        label="Validation Log Loss",
    )
    ax1.axvline(
        5,
        color="#27ae60",
        linestyle="--",
        linewidth=1.8,
        label="Chosen: 5 Bins (Log Loss = 0.9766)",
    )
    ax1.scatter([5], [0.9766], color="#27ae60", s=90, zorder=5)
    ax1.set_title("Naive Bayes: Quantile Bin Tuning (4 Features)")
    ax1.set_xlabel("Number of Quantile Bins (NB_BIN_GRID)")
    ax1.set_ylabel("Validation Log Loss (Lower is Better)")
    ax1.set_xticks(nb_bins)
    ax1.legend(loc="upper right", frameon=True)

    # Right: Logistic Regression C
    ax2.plot(
        lr_c,
        lr_logloss,
        marker="s",
        linewidth=2.2,
        color="#2980b9",
        label="Validation Log Loss",
    )
    ax2.axvline(
        0.01,
        color="#27ae60",
        linestyle="--",
        linewidth=1.8,
        label="Chosen: C = 0.01 (Log Loss = 0.9677)",
    )
    ax2.scatter([0.01], [0.9677], color="#27ae60", s=90, zorder=5)
    ax2.set_xscale("log")
    ax2.set_title("Logistic Regression: Regularization Tuning (11 Features)")
    ax2.set_xlabel("Inverse Regularization Strength C (Log Scale)")
    ax2.set_ylabel("Validation Log Loss (Lower is Better)")
    ax2.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    out = REPORTS_DIR / "plot_3_hyperparameter_tuning.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_4_lr_coefficients():
    """Plot standardized Logistic Regression coefficients across Away, Draw, and Home."""
    coef_file = REPORTS_DIR / "logistic_regression_coefficients.csv"
    if coef_file.exists():
        df = pd.read_csv(coef_file)
        if "feature" not in df.columns:
            df = df.rename(columns={df.columns[0]: "feature"})
    else:
        df = pd.DataFrame(
            [
                ("elo_diff", -0.378, -0.033, 0.411),
                ("form5_diff", 0.070, 0.019, -0.090),
                ("venue_ppg_diff", -0.071, 0.010, 0.061),
                ("sot5_diff", -0.139, 0.024, 0.115),
                ("ga5_diff", 0.024, 0.012, -0.035),
                ("gf5_diff", 0.041, -0.046, 0.005),
                ("h2h_pts", -0.048, 0.052, -0.005),
                ("abs_elo_diff", 0.033, -0.069, 0.036),
                ("matchweek", 0.002, -0.021, 0.019),
                ("away_promoted", -0.019, -0.019, 0.037),
                ("draw_rate_home", -0.003, -0.005, 0.007),
            ],
            columns=["feature", "coef_A", "coef_D", "coef_H"],
        )

    # Order by Home Win coefficient magnitude
    df = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(df))
    height = 0.26

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(
        y - height, df["coef_A"], height, label="Away Win (coef_A)", color="#e74c3c"
    )
    ax.barh(y, df["coef_D"], height, label="Draw (coef_D)", color="#95a5a6")
    ax.barh(
        y + height, df["coef_H"], height, label="Home Win (coef_H)", color="#2980b9"
    )

    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"])
    ax.set_xlabel("Standardized Coefficient Weight (C = 0.01)")
    ax.set_title(
        "Multinomial Logistic Regression: Standardized Coefficients per Outcome"
    )
    ax.legend(frameon=True, loc="lower right")

    fig.tight_layout()
    out = REPORTS_DIR / "plot_4_lr_coefficients.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_5_model_comparison():
    """Plot Test Set evaluation metrics comparing models against Baselines & Bet365."""
    models = [
        "Always Home\n(Baseline)",
        "Naive Bayes\n(All 19 Feats)",
        "Naive Bayes\n(4 Selected)",
        "Logistic Reg.\n(11 Selected)",
        "Bet365 Odds\n(Bookmaker)",
    ]
    accuracy = [45.35, 50.88, 52.98, 54.12, 54.12]
    log_loss = [1.0669, 1.5502, 0.9797, 0.9699, 0.9624]
    brier = [0.6447, 0.7177, 0.5834, 0.5756, 0.5709]
    rps = [0.2275, 0.2534, 0.1983, 0.1946, 0.1923]

    colors = ["#95a5a6", "#e67e22", "#3498db", "#2ecc71", "#9b59b6"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    metrics_info = [
        (
            axes[0, 0],
            accuracy,
            "Test Accuracy (%) — Higher is Better",
            "{:.2f}%",
            (40, 58),
        ),
        (
            axes[0, 1],
            log_loss,
            "Test Log Loss — Lower is Better",
            "{:.4f}",
            (0.85, 1.68),
        ),
        (
            axes[1, 0],
            brier,
            "Test Brier Score — Lower is Better",
            "{:.4f}",
            (0.50, 0.76),
        ),
        (
            axes[1, 1],
            rps,
            "Ranked Probability Score (RPS) — Lower is Better",
            "{:.4f}",
            (0.17, 0.27),
        ),
    ]

    for ax, values, title, fmt, ylim in metrics_info:
        bars = ax.bar(
            models, values, color=colors, edgecolor="black", linewidth=0.6, width=0.58
        )
        ax.set_title(title)
        ax.set_ylim(*ylim)
        for bar, val in zip(bars, values):
            ax.annotate(
                fmt.format(val),
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    fig.suptitle(
        "Test Set Evaluation (1,140 Matches: 2015–16 to 2017–18)",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out = REPORTS_DIR / "plot_5_model_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_6_confusion_matrices():
    """Plot side-by-side Confusion Matrices for Naive Bayes and Logistic Regression."""
    nb_cm = np.array(
        [
            [178, 0, 155],
            [95, 0, 195],
            [91, 0, 426],
        ]
    )
    lr_cm = np.array(
        [
            [167, 0, 166],
            [71, 0, 219],
            [67, 0, 450],
        ]
    )

    labels = ["Away (A)", "Draw (D)", "Home (H)"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    matrices = [
        (axes[0], nb_cm, "Naive Bayes (4 Features, 5 Bins)\nTest Accuracy: 52.98%"),
        (
            axes[1],
            lr_cm,
            "Logistic Regression (11 Features, C=0.01)\nTest Accuracy: 54.12%",
        ),
    ]

    for ax, cm, title in matrices:
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=460)
        ax.set_title(title)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels([f"Pred {l}" for l in labels])
        ax.set_yticklabels([f"True {l}" for l in labels])
        ax.grid(False)

        row_sums = cm.sum(axis=1, keepdims=True)
        for i in range(3):
            for j in range(3):
                val = cm[i, j]
                pct = (val / row_sums[i, 0]) * 100
                text_color = "white" if val > 220 else "black"
                ax.text(
                    j,
                    i,
                    f"{val}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10.5,
                    fontweight="bold",
                )

    fig.suptitle(
        "Test Set Confusion Matrices (Illustrating the Argmax Zero-Draw Effect)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out = REPORTS_DIR / "plot_6_confusion_matrices.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating presentation charts into reports/ ...")
    plot_1_class_balance()
    plot_2_mutual_information()
    plot_3_hyperparameter_tuning()
    plot_4_lr_coefficients()
    plot_5_model_comparison()
    plot_6_confusion_matrices()
    print("All 6 presentation graphs generated successfully!")
