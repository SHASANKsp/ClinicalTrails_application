class ClinicalPortfolioMetrics:

    def __init__(self, df):
        self.df = df.copy()

    def total_trials(self):
        return self.df["nct_id"].nunique()

    def condition_diversity(self):
        return self.df["condition"].nunique()

    def sponsor_diversity(self):
        return self.df["sponsor"].nunique()

    def geographic_spread(self):
        return self.df["country"].nunique()

    def completion_ratio(self):
        return (self.df["status"] == "COMPLETED").mean()

    def phase_maturity_score(self):
        return (
            self.df["phase"]
            .astype(str)
            .str.contains("3", na=False)
            .mean()
        )

    def avg_enrollment(self):
        return self.df["enrollment"].mean()

    def arm_complexity(self):
        return (
            self.df.groupby("nct_id")["arm_label"]
            .nunique()
            .mean()
        )

    def repurposing_strength(self):
        return (
            self.condition_diversity()
            * self.phase_maturity_score()
            * self.completion_ratio()
        )