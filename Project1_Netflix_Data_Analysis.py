"""
PROJECT 1: NETFLIX TITLES — EXPLORATORY DATA ANALYSIS (EDA)
Student: Niresh
Platform: Google Colab / Python

Dataset:
Kaggle - Netflix Movies and TV Shows
https://www.kaggle.com/datasets/shivamb/netflix-shows

Expected CSV filename: netflix_titles.csv
This is a descriptive data analysis project, not a prediction model.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------
DATA_FILE = "netflix_titles.csv"

if not os.path.exists(DATA_FILE):
    # Helpful for Google Colab: ask user to upload the CSV if missing.
    try:
        from google.colab import files
        print("Please select the netflix_titles.csv file downloaded from Kaggle.")
        uploaded = files.upload()
        if DATA_FILE not in uploaded:
            raise FileNotFoundError(
                f"Expected '{DATA_FILE}'. Rename the downloaded dataset to "
                f"'{DATA_FILE}' and run again."
            )
    except ImportError:
        raise FileNotFoundError(
            "Place netflix_titles.csv in the same folder as this script."
        )

df = pd.read_csv(DATA_FILE)

required_cols = {
    "show_id", "type", "title", "director", "cast", "country",
    "date_added", "release_year", "rating", "duration",
    "listed_in", "description"
}
missing_cols = required_cols.difference(df.columns)
if missing_cols:
    raise ValueError(
        "Dataset is missing expected columns: " + ", ".join(sorted(missing_cols))
    )

print("=" * 65)
print("NETFLIX TITLES — EXPLORATORY DATA ANALYSIS")
print("=" * 65)
print("Rows and columns before cleaning:", df.shape)
print("\nFirst five rows:")
print(df.head().to_string(index=False))
print("\nData types and non-null counts:")
df.info()
print("\nMissing values before cleaning:")
print(df.isnull().sum().to_string())
print("\nDuplicate rows before cleaning:", df.duplicated().sum())

# ------------------------------------------------------------
# 2. CLEAN AND PREPARE DATA
# ------------------------------------------------------------
df = df.drop_duplicates().copy()

# Invalid or unrecognized dates become NaT; they are not guessed.
df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")

text_cols = [
    "director", "cast", "country", "rating",
    "duration", "listed_in", "description"
]
for col in text_cols:
    df[col] = df[col].fillna("Unknown").astype(str).str.strip()
    df.loc[df[col].eq(""), col] = "Unknown"

df["year_added"] = df["date_added"].dt.year
df["duration_number"] = pd.to_numeric(
    df["duration"].str.extract(r"(\d+)")[0], errors="coerce"
)

print("\n" + "=" * 65)
print("DATA QUALITY AFTER CLEANING")
print("=" * 65)
print("Shape after cleaning:", df.shape)
print("Duplicates remaining:", df.duplicated().sum())
print("\nMissing values after cleaning:")
print(df.isnull().sum().to_string())

# ------------------------------------------------------------
# 3. ANSWER FIVE DATA QUESTIONS
# ------------------------------------------------------------
print("\nQ1 — Movies vs TV Shows")
type_counts = df["type"].value_counts()
print(type_counts.to_string())

print("\nQ2 — Top countries (titles can list more than one country)")
countries = df.loc[df["country"].ne("Unknown"), "country"].str.split(", ").explode()
country_counts = countries.value_counts().head(10)
print(country_counts.to_string() if not country_counts.empty else "No country data available.")

print("\nQ3 — Top content ratings")
rating_counts = df.loc[df["rating"].ne("Unknown"), "rating"].value_counts().head(10)
print(rating_counts.to_string() if not rating_counts.empty else "No rating data available.")

print("\nQ4 — Top categories / genres (a title can have multiple categories)")
genres = df.loc[df["listed_in"].ne("Unknown"), "listed_in"].str.split(", ").explode()
genre_counts = genres.value_counts().head(10)
print(genre_counts.to_string() if not genre_counts.empty else "No category data available.")

print("\nQ5 — Titles added by year")
year_counts = df["year_added"].dropna().astype(int).value_counts().sort_index()
print(year_counts.to_string() if not year_counts.empty else "No valid date-added values available.")

if not type_counts.empty:
    print("\nLargest content type count:", type_counts.idxmax(), int(type_counts.max()))
if not country_counts.empty:
    print("Most frequently listed country:", country_counts.index[0], int(country_counts.iloc[0]))
if not rating_counts.empty:
    print("Most common known rating:", rating_counts.index[0], int(rating_counts.iloc[0]))
if not genre_counts.empty:
    print("Most frequent category:", genre_counts.index[0], int(genre_counts.iloc[0]))
if not year_counts.empty:
    print("Year with the most titles added:", int(year_counts.idxmax()), int(year_counts.max()))

# ------------------------------------------------------------
# 4. SIX VISUALIZATIONS
# ------------------------------------------------------------
sns.set_theme()  # Use Seaborn's default theme for readable plots.

# 1. Bar chart: Movies vs TV Shows
plt.figure(figsize=(7, 4))
type_counts.plot(kind="bar", title="Movies vs TV Shows")
plt.xlabel("Content type")
plt.ylabel("Number of titles")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 2. Line chart: titles added by year
if not year_counts.empty:
    plt.figure(figsize=(8, 4))
    year_counts.plot(kind="line", marker="o", title="Titles Added to Netflix by Year")
    plt.xlabel("Year added")
    plt.ylabel("Number of titles")
    plt.tight_layout()
    plt.show()

# 3. Histogram: release-year distribution
plt.figure(figsize=(8, 4))
df["release_year"].dropna().plot(
    kind="hist", bins=20, title="Distribution of Release Years"
)
plt.xlabel("Release year")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4. Scatter plot: release year vs year added
tmp = df.dropna(subset=["release_year", "year_added"])
plt.figure(figsize=(7, 4))
plt.scatter(tmp["release_year"], tmp["year_added"], alpha=0.35)
plt.title("Release Year vs Year Added")
plt.xlabel("Release year")
plt.ylabel("Year added")
plt.tight_layout()
plt.show()

# 5. Pie chart: Movies vs TV Shows share
if not type_counts.empty:
    type_counts.plot(
        kind="pie", autopct="%1.1f%%",
        title="Share of Movies and TV Shows", figsize=(6, 6)
    )
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

# 6. Heatmap: numeric correlations
corr_cols = ["release_year", "year_added", "duration_number"]
plt.figure(figsize=(6, 4))
sns.heatmap(
    df[corr_cols].corr(numeric_only=True),
    annot=True, cmap="coolwarm", vmin=-1, vmax=1
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. SAVE CLEANED DATA AND SUMMARY OUTPUTS
# ------------------------------------------------------------
df.to_csv("netflix_titles_cleaned.csv", index=False)
print("\nSaved cleaned dataset as: netflix_titles_cleaned.csv")

summary = pd.DataFrame({
    "Content Type": type_counts.index,
    "Count": type_counts.values
})
summary.to_csv("netflix_type_summary.csv", index=False)
print("Saved content-type summary as: netflix_type_summary.csv")

print("\nPROJECT COMPLETED.")
print("Use the real values printed above in your report; do not invent findings.")
print("Note: This project describes the dataset and does not prove why Netflix made a decision.")
