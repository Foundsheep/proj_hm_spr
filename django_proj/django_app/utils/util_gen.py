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
        self.is_javascript = False
        self.method = self._give_unique_list_by_key if self.is_javascript else self._give_unique_tupled_list_by_key
        
    def get_condition_options_rivet(self):
        return self.method(self.rivet)
        
    def get_condition_options_die(self):
        return self.method(self.die)

    def get_condition_options_upper_type(self):
        return self.method(self.upper_type)
        
    def get_condition_options_middle_type(self):
        return self.method(self.middle_type)

    def get_condition_options_lower_type(self):
        return self.method(self.lower_type)
        
    def get_dummy_df(self, path):
        return pd.read_csv(path)

    def _give_unique_list_by_key(self, key):
        return self.df[key].dropna().unique().tolist()
    
    def _give_unique_dict_by_key(self, key):
        return {c: c for c in self.df[key].dropna().unique()}
    
    def _give_unique_tupled_list_by_key(self, key):
        return [(c, c) for c in self.df[key].dropna().unique()]