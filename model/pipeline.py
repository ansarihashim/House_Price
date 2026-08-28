"""Inference-time pipeline object mirroring notebooks/House_PRICE.ipynb (cells 10-12).

Bundles the fitted Gradient Boosting model together with the winsorization
bounds and feature order it was trained with, so both travel as one
serialized artifact and can never drift apart.
"""


class HousePriceModel:
    def __init__(self, winsorize_bounds, feature_order, model):
        self.winsorize_bounds = winsorize_bounds
        self.feature_order = feature_order
        self.model = model

    def engineer_features(self, df):
        df = df.copy()

        for col, (lower, upper) in self.winsorize_bounds.items():
            df[col] = df[col].clip(lower, upper)

        df["RoomsPerBedroom"] = df["AveRooms"] / df["AveBedrms"].replace(0, 1)
        df["BedroomRatio"] = df["AveBedrms"] / df["AveRooms"].replace(0, 1)
        df["PopulationPerHousehold"] = df["Population"] / df["AveOccup"].replace(0, 1)
        df["MedIncSq"] = df["MedInc"] ** 2

        return df[self.feature_order]

    def predict(self, df):
        engineered = self.engineer_features(df)
        return self.model.predict(engineered)
