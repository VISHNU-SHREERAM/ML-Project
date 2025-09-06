from sklearn.decomposition import PCA
import pandas as pd

def do_pca(df):
    features = [
        key for key in df if (key != "Sleep Disorder" and key != "Stress Level")
    ]

    # ['Insomnia', 'None', 'Sleep Apnea'] the labels associated with 0,1,2
    pca = PCA(n_components=len(features))
    df_full_pca = pca.fit_transform(df[features])

    return pca, df_full_pca


def get_new_components(df):
    features = [
        key for key in df if (key != "Sleep Disorder" and key != "Stress Level")
    ]
    new_pca = PCA(n_components=5)
    new_pca.fit(df[features])

    new_points = new_pca.transform(df[features])
    return new_pca, new_points


def get_top_features(model, feature_names, top_n=10):
    feat_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })
    return feat_df.sort_values("importance", ascending=False).head(top_n)