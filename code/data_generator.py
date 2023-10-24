import torch
import torch_geometric
import torch_geometric.transforms as T
from glob import glob
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cftime
import pickle

class ssp_data():
    def __init__(self, n=39) -> None:
        self.n = n
        self.init_edge_list(n)
        self.y_file = 'data\\tas_scenario_245\\tas_mon_mod_ssp245_192_000.nc'
        self.x_file_list = [item for item in glob('data\\tas_scenario_245\\tas_mon_mod_ssp245_192_*.nc') if item not in [self.y_file]][0 : self.n]
        self.create_df()
        self.test_train_split()
       
        # self.split_data()
        # self.mini_graphs()

    def init_edge_list(self, n):
        self.edge_index = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.edge_index.append([i, j])
        self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)

    def create_df(self):
        self.x = pd.DataFrame()
        i = 1
        for filename in self.x_file_list:
            print('Processing', filename)
            if self.x.empty:
                self.x = self.create_vector(filename).reset_index(drop=True)
            else:
                assert len(self.x) == len(w:= self.create_vector(filename)['tas'])
                self.x[f'tas_{i}'] = w.reset_index(drop=True)
                # self.x = self.x.merge(self.create_vector(filename), how='inner', on=['time', 'lat', 'lon'], suffixes=(None, f'_{i}'))
            
            i += 1
        
        self.y = self.create_vector(self.y_file)['tas'].reset_index(drop=True)

    def create_vector(self, filename):
        data = xr.open_dataset(filename)
        try:
            datetimeindex = data.indexes['time'].to_datetimeindex()
            data['time'] = datetimeindex
        except AttributeError:
            pass

        df = data.to_dataframe().reset_index()
        # df = df.query('lat == -43.125 & lat == 288.750')
        df = df.query('lat >= -44 & lat <= -12 & lon >= 288 & lon <= 336')
        ret = df.loc[(df['time'].dt.year > 1960) & (df['time'].dt.year < 1980), ['time', 'lat', 'lon', 'tas']]

        return ret
    
    def test_train_split(self, p=74100):
        df = self.x.drop(columns=['time', 'lat', 'lon'], axis=1)
        self.x_train = df[0:p]
        self.x_test = df[p:]
        self.y_train = df[0:p]
        self.y_test = df[p:]

        self.x_train = self.create_tensors(self.x_train)
        self.y_train = self.create_tensors(self.y_train)
        self.train_data = torch_geometric.data.Data(x=self.x_train, edge_index=self.edge_index.t().contiguous(), y=self.y_train)

        self.x_test = self.create_tensors(self.x_test)
        self.y_test = self.create_tensors(self.y_test)
        self.test_data = torch_geometric.data.Data(x=self.x_test, edge_index=self.edge_index.t().contiguous(), y=self.y_test)

    
    def get_device(self):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu') # don't have GPU 
        return device

    def create_tensors(self, df):
        device = self.get_device()
        return torch.from_numpy(df.values).transpose(0, 1).float().to(device)
    
    def to_pickle(self, name='data_pickle'):
        file = open(name, 'wb')
        pickle.dump(self, file)
        file.close()
    
    # def mini_graphs(self):
    #     df = self.x
    #     df['x_tensor'] = df.apply(lambda row: torch.tensor(row.values.flatten()), axis=1)
    #     df['y'] = self.y
    #     df['y_tensor'] = df['y'].apply(lambda y: torch.tensor(y))
    #     df['data_obj'] = df.apply(lambda row: torch_geometric.data.Data(x=df['x_tensor'], edge_index=self.edge_index.t().contiguous(), y=df['y_tensor']), axis=1)
    #     self.batch_graphs = df['data_obj']
    
    # def split_data(self):
    #     transform = T.Compose([T.RandomNodeSplit(num_test=1000, num_val=1000)])
    #     self.data = transform(self.data) 
