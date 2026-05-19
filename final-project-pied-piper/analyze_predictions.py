import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("rand_test_predictions.csv")

    print("\nBasic Info:")
    print(f"Total samples: {len(df)}")

   
    print("\nClass Distribution:")
    counts = df["prediction"].value_counts()
    print(counts)

    print("\nPercentage:")
    print(counts / len(df))

    
    print("\nConfidence Statistics:")
    print(df["fake_probability"].describe())

    print("\nTop 10 most confident fake predictions:")
    print(df.sort_values("fake_probability", ascending=False).head(10))

    
    print("\nTop 10 most confident real predictions:")
    print(df.sort_values("fake_probability", ascending=True).head(10))

   
    print("\nMost uncertain predictions (closest to 0.5):")
    df["uncertainty"] = (df["fake_probability"] - 0.5).abs()
    print(df.sort_values("uncertainty").head(10))

    
    plt.figure()
    plt.hist(df["fake_probability"], bins=50)
    plt.title("Fake Probability Distribution")
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.show()

    
    summary = {
        "total_samples": len(df),
        "num_fake_predicted": int((df["prediction"] == 1).sum()),
        "num_real_predicted": int((df["prediction"] == 0).sum()),
        "mean_probability": float(df["fake_probability"].mean()),
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("rand_test_summary.csv", index=False)

    print("\nSummary saved to rand_test_summary.csv")


if __name__ == "__main__":
    main()