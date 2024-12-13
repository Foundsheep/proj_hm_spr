import pandas as pd

class GenOptionProducer:
    def __init__(self, path):
        self.df = self.get_dummy_df(path)
        self.rivet = "rivet"
        self.die = "die"
        self.upper_type = "upper_type"
        self.upper_thickness = "upper_thickness"
        self.middle_type = "middle_type"
        self.middle_thickness = "middle_thickness"
        self.lower_type = "lower_type"
        self.lower_thickness = "lower_thickness"
        
    def get_condition_options_upper_type(self):
        return self._give_unique_list_by_key(self.upper_type)
        
    def get_dummy_df(self, path):
        return pd.read_csv(path)

    def _give_unique_list_by_key(self, key):
        return self.df[key].dropna().unique().tolist()