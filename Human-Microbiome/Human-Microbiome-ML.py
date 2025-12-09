import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np

# Load data
df = pd.read_csv(r"C:\Users\AliSolaxay\OneDrive\Desktop\AITest\Human Microbiome.csv")

df_selected = df[[
    "Organism Name",
    "Domain",
    "NCBI Superkingdom",
    "HMP Isolation Body Site"
]].copy()

df_selected = df_selected.fillna("Unknown")

# Column transformer
categorical_cols = df_selected.columns.tolist()

preprocess = ColumnTransformer([
    ("cat",
     OneHotEncoder(handle_unknown="ignore", sparse_output=False),
     categorical_cols)
])

# K-Fold setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Search range
cluster_range = range(2, 8)
grid_results = {}
results_list = []


# Manual GridSearch
for k in cluster_range:
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("pca", PCA(n_components=2)),
        ("kmeans", KMeans(n_clusters=k, random_state=42))
    ])

    scores = []
    for train_idx, test_idx in kf.split(df_selected):
        train = df_selected.iloc[train_idx]
        test = df_selected.iloc[test_idx]

        pipe.fit(train)
        clusters = pipe.predict(test)

        transformed = pipe.named_steps["pca"].transform(
            pipe.named_steps["preprocess"].transform(test)
        )

        score = silhouette_score(transformed, clusters)
        scores.append(score)

    avg_score = np.mean(scores)
    grid_results[k] = avg_score
    results_list.append({"k": k, "silhouette": avg_score})


# Save results to CSV
results_df = pd.DataFrame(results_list)
results_df.to_csv("microbiome_gridsearch_results.csv", index=False)
print("\nSaved to -> microbiome_gridsearch_results.csv")

# Best K
best_k = results_df.loc[results_df["silhouette"].idxmax(), "k"]
best_score = results_df["silhouette"].max()
print("\nBest K:")
print(f"K = {best_k}, Silhouette = {best_score:.4f}")

# Print all results
print("\nAll results:")
print(results_df)

# Heatmap
encoded = preprocess.fit_transform(df_selected)
encoded_df = pd.DataFrame(encoded)

plt.figure(figsize=(12, 10))
sns.heatmap(pd.DataFrame(encoded_df).corr(), cmap="viridis")
plt.title("Encoded Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
