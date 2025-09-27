import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Define dataset
data = {
    "Country": [
        "Turkey","Azerbaijan","South Azerbaijan(Tabriz)","Kazakhstan",
        "Kyrgyzstan","Turkmenistan","Uzbekistan","Tatarstan",
        "Crimea(Simferopol)","Iraq(Kirkuk)","East Turkistan"
    ],
    "Allergy_Type": ["Pollen","Dust","Food","Pollen","Dust","Food","Drug","Pollen",
                     "Dust","Food","Pollen"],
    "Prevalence": [20, 15, 17, 12, 18, 14, 10, 8, 11, 13, 22],
    "Lat": [39.9334, 40.4093, 38.0700, 51.1694, 42.8746, 38.9637,
            41.3775, 55.7963, 44.9521, 35.4667, 43.8256],
    "Lon": [32.8597, 49.8671, 46.2919, 71.4491, 74.6122, 59.5563,
            64.5853, 49.1064, 34.1024, 44.4009, 87.6168]
}

df = pd.DataFrame(data)

# Save dataset to CSV
csv_file = "Turkic_Allergy_Data_Expanded.csv"
df.to_csv(csv_file, index=False, encoding="utf-8")

print(f" CSV file saved as: {csv_file}")
df

fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

for _, row in df.iterrows():
    ax.scatter(row["Lon"], row["Lat"],
               s=row["Prevalence"] * 20,
               c=row["Prevalence"],
               cmap="Reds",
               alpha=0.6,
               transform=ccrs.PlateCarree())
    ax.text(row["Lon"] + 1, row["Lat"] + 0.5,
            row["Country"] + f" ({row['Prevalence']}%)",
            fontsize=8,
            transform=ccrs.PlateCarree())

plt.title("Allergy Prevalence Among Turkic Regions (Expanded)", fontsize=14)
plt.colorbar(plt.cm.ScalarMappable(cmap="Reds"),
             ax=ax, orientation="vertical", label="Prevalence (%)")
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x="Country", y="Prevalence", hue="Allergy_Type", data=df)
plt.xticks(rotation=45, ha="right")
plt.title("Prevalence of Allergies by Type in Turkic Regions (Expanded)")
plt.tight_layout()
plt.show()
