from load_data import DataLoader


class Opts:
    n_layer = 2
    hidden_dim = 128
    perf_file = 'perf.txt'


opts = Opts()
loader = DataLoader(
    'F:\Courses\大三下\生物信息学\SL_Project_Baseline\KR4SL\data\\transductive', opts)
