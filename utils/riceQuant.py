import rqdatac
rqdatac.init()

import numpy as np
import pandas as pd
import os
import copy
from collections import Counter


class RiceQuantExtractor:
    def __init__(self) -> None:
        # CSI300 index, CSI300 constituent stocks
        # Stock data
        self.fdir = './data'
        os.makedirs(self.fdir, exist_ok=True)

        # 数据下载时间范围
        self.start_date = '2013-08-30'  # 数据下载开始日期
        self.end_date = '2023-09-02'    # 数据下载结束日期
        
        # 权重基准日期（用于确定选择哪些股票）
        self.weight_date = '2013-08-31'  # 权重基准日期，可以独立设置
        
        self.market_name = 'CSI300'
        self.market_code = '000300.XSHG'
        self.freq = '60m' # '1d', '60m', 'tick'
        self.topK = 30
        self.request_fields = ['open', 'high', 'low', 'close', 'volume'] # 'total_turnover']
         
    def get_price_data(self, mkt_code, tic, asset_name, recidx, suffix=None, start_date=None, end_date=None):
        # Get the stock data 	
        print("*"*30)
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
        mkt_index = rqdatac.get_price(order_book_ids=[mkt_code], start_date=start_date, end_date=end_date, frequency=self.freq, fields=self.request_fields, skip_suspended=False, market='cn')
        mkt_index = pd.DataFrame(mkt_index)
        if len(mkt_index) == 0:
            print("Cannot find the data between {} and {} for code: {}".format(start_date, end_date, mkt_code))
            return 0
        mkt_index.reset_index(drop=False, inplace=True)
        mkt_index.rename(columns={'datetime': 'date'}, inplace=True)
        mkt_index['date'] = pd.to_datetime(mkt_index['date'])
        mkt_index.sort_values(by=['date'], ascending=True, inplace=True, ignore_index=True)
        mkt_index['tic'] = tic
        # print(mkt_index)

        mkt_cols = ['date', 'tic'] + list(self.request_fields)
        mkt_index = mkt_index[mkt_cols]
        
        null_num = mkt_index.isnull().sum().sum()
        if null_num > 0:
            print("NULL values are found at: ")
            null_num_series = mkt_index.isnull().sum()
            print(null_num_series[null_num_series != 0])

        na_num = mkt_index.isna().sum().sum()
        if na_num > 0:
            print("NA values are found at: ")
            na_num_series = mkt_index.isna().sum()
            print(na_num_series[na_num_series != 0])  

        close_previous = np.array(copy.deepcopy(mkt_index['close']))
        close_previous = np.append([close_previous[0]], close_previous[:-1], axis=0) 
        for idxf in ['open', 'high', 'low', 'close', 'volume']:
            fdata = np.array(copy.deepcopy(mkt_index[idxf]))

            # Checking for any abnormal data
            l0 = len(fdata[fdata < 0])
            if l0 > 0:
                print('Field {} has {} negative records.'.format(idxf, l0))  
                print("Index: {}".format(np.array(mkt_index[fdata < 0].index)))
                print("-"*10)

            l0x = len(fdata[fdata == 0])
            if l0x > 0:
                print('Field {} has {} zero records.'.format(idxf, l0x))  
                print("Index: {}".format(np.array(mkt_index[fdata == 0].index)))
                print("-"*10)

            if idxf == 'volume':
                l1 = len(fdata[fdata >= 1e10])
                if l1 > 0:
                    print('Field {} has {} records larger than 1e10.'.format(idxf, l1))
                    print("Index: {}".format(np.array(mkt_index[fdata >= 1e10].index)))
                    print("-"*10)
                
                fdata[fdata <= 0] = 1
                dx = np.diff(fdata)
                dy = dx/fdata[:-1]
                l2 = len(dy[dy > 5])
                if l2 > 0:
                    fxdata = np.append([False], dy > 5, axis=0)
                    print('Field {} has {} records larger than 5x change rate.'.format(idxf, l2))
                    print("Index: {}".format(np.array(mkt_index[fxdata].index)))
                    print("-"*10)
                l3 = len(dy[dy < -0.5])
                if l3 > 0:
                    fxdata = np.append([False], dy < -0.5, axis=0)
                    print('Field {} has {} records smaller than -0.5x change rate.'.format(idxf, l3))
                    print("Index: {}".format(np.array(mkt_index[fxdata].index)))
                    print("-"*10)
            else:
                l1 = len(fdata[fdata > 1e4])
                if l1 > 0:
                    print('Field {} has {} records larger than 1e4.'.format(idxf, l1))
                    print("Index: {}".format(np.array(mkt_index[fdata > 1e4].index)))
                    print("-"*10)

                dy = (fdata - close_previous) / close_previous
                dy = dy[1:]
                l2 = len(dy[dy > 0.11])
                if l2 > 0:
                    fxdata = np.append([False], dy > 0.11, axis=0)
                    print('Field {} has {} records larger than 0.11 change rate.'.format(idxf, l2))
                    print("Index: {}".format(np.array(mkt_index[fxdata].index)))
                    print("-"*10)
        print("-"*20)
        
        if len(mkt_index) != mkt_index['date'].nunique():
            duplicated_date = Counter(mkt_index['date'])
            duplicated_date = pd.DataFrame({'date': duplicated_date.keys(), 'count': duplicated_date.values()})
            print("Duplicated date is found at: ")
            print(duplicated_date[duplicated_date['count'] > 1])
        
        mkt_index.sort_values(by=['date'], ascending=True, inplace=True, ignore_index=True)
        mkt_index = copy.deepcopy(mkt_index)
        last_vol = mkt_index['volume'][0]
        for idx in range(1, len(mkt_index)):
            if mkt_index['volume'][idx] <= 0:
                mkt_index.loc[idx, 'volume'] = last_vol
            last_vol = mkt_index['volume'][idx]
        
        if np.sum(mkt_index['volume'] <= 0) > 0:
            print('Volume has {} records smaller than 0.'.format(np.array(mkt_index[mkt_index['volume'] <= 0].index)))
            print("-"*10)

        if suffix is None:
            # stock
            fxdir = os.path.join(self.fdir, '{}_{}_{}'.format(self.market_name, self.topK, self.freq))
            os.makedirs(fxdir, exist_ok=True)
            fpath = os.path.join(fxdir, '{}_{}.csv'.format(mkt_code, self.freq))
        else:
            # market index
            fpath = os.path.join(self.fdir, '{}_{}_{}.csv'.format(self.market_name, self.freq, suffix))
        if os.path.exists(fpath):
            raw_data = pd.DataFrame(pd.read_csv(fpath, header=0))
            raw_data['date'] = pd.to_datetime(raw_data['date'])
            raw_data = pd.concat([raw_data, mkt_index], axis=0, ignore_index=True)
            raw_data.drop_duplicates(subset=['date'], keep='last', inplace=True, ignore_index=True)
            raw_data.sort_values(by=['date'], ascending=True, inplace=True, ignore_index=True)
            raw_data.to_csv(fpath, index=False)
        else:
            mkt_index.drop_duplicates(subset=['date'], keep='last', inplace=True, ignore_index=True)
            mkt_index.sort_values(by=['date'], ascending=True, inplace=True, ignore_index=True)
            mkt_index.to_csv(fpath, index=False)
    
        print("Done at code: {}, timepoints: {}, tic: {}, asset_name: {}, idx: {}..".format(mkt_code, mkt_index['date'].nunique(), tic, asset_name, recidx))


    def get_market_index_data(self):
        mkt_code = self.market_code
        tic = self.market_name
        asset_name = self.market_name
        recidx = -1
        self.get_price_data(mkt_code=mkt_code, tic=tic, asset_name=asset_name, recidx=recidx, suffix='index')
    
    def get_stock_price_data(self):
        # Get constituent stock list in CSI300 OR you may download specific stock data by using the API provided by RiceQuant.
        wpath = os.path.join(self.fdir, 'Weights', '{}_Weights_{}.csv'.format(self.market_name, self.freq))
        if not os.path.exists(wpath):
            raise ValueError("Please provide a list of constituent stocks")

        weights = pd.read_csv(wpath) 
        weights.sort_values(by=['rank'], ascending=True, inplace=True, ignore_index=True)
        total_num_stocks = len(weights)
        cnt1 = 0
        cnt2 = 0

        for idx in range(total_num_stocks):
            fxdir = os.path.join(self.fdir, '{}_{}_{}'.format(self.market_name, self.topK, self.freq))
            fpath = os.path.join(fxdir, '{}_{}.csv'.format(weights['code'][idx], self.freq))
            if os.path.exists(fpath):
                cnt1 = cnt1 + 1
                print("The data file of stock {} already exists.".format(weights['code'][idx]))
            else:
                cnt2 = cnt2 + 1
                self.get_price_data(mkt_code=weights['code'][idx], tic=weights['code'][idx], 
                                asset_name=weights['code'][idx], recidx=idx)
        # print("cnt1: {}, cnt2: {}".format(cnt1, cnt2))
    
    def get_constituent_by_weight(self):
        # 根据权重获取CSI300指数前topK支股票
        print(f"获取CSI300指数权重数据 (权重基准日期: {self.weight_date})")
        print(f"数据下载时间范围: {self.start_date} 到 {self.end_date}")
        
        # 获取所有成分股的权重数据
        all_weights = rqdatac.index_weights(order_book_id=self.market_code, date=self.weight_date)
        all_weights = pd.DataFrame(all_weights)
        all_weights.reset_index(drop=False, inplace=True)
        all_weights.rename(columns={'order_book_id': 'code', 0: 'weight'}, inplace=True)
        
        print(f"获取到CSI300成分股数量: {len(all_weights)}")
        
        # 按权重降序排列
        all_weights.sort_values(by=['weight'], ascending=False, inplace=True, ignore_index=True)
        
        # 选择前topK支股票
        cap_weight = all_weights.head(self.topK).copy()
        cap_weight['rank'] = np.arange(1, len(cap_weight)+1)
        
        print(f"按权重排序，选择前{self.topK}支股票:")
        print(f"权重前10名股票: {all_weights.head(10)['code'].tolist()}")
        print(f"选中的前{self.topK}支股票:")
        print(cap_weight)

        # 保存权重文件
        fdirx = os.path.join(self.fdir, 'Weights')  
        os.makedirs(fdirx, exist_ok=True)
        fpath = os.path.join(fdirx, '{}_Weights_{}.csv'.format(self.market_name, self.freq))
        cap_weight.to_csv(fpath, index=False)
        print(f"权重文件已保存到: {fpath}")

    def demo(self):
        mkt_index = rqdatac.get_price(order_book_ids=['000300.XSHG'], start_date='2024-01-01', end_date='2024-10-31', frequency='1d', fields=self.request_fields, skip_suspended=False, market='cn')
        print(mkt_index)

    def merge_stock_data(self):
        fdir = os.path.join(self.fdir, '{}_{}_{}'.format(self.market_name, self.topK, self.freq))
        if not os.path.exists(fdir):
            raise ValueError("Please provide a list of constituent stocks")
        
        # 读取权重文件，按权重排序来确定股票编号
        wpath = os.path.join(self.fdir, 'Weights', '{}_Weights_{}.csv'.format(self.market_name, self.freq))
        weights = pd.read_csv(wpath)
        weights.sort_values(by=['rank'], ascending=True, inplace=True, ignore_index=True)
        
        all_data = []
        
        # 按权重文件中的顺序处理股票数据
        for idx, row in weights.iterrows():
            stock_code = row['code']
            stock_rank = row['rank']
            file_path = os.path.join(fdir, f'{stock_code}_{self.freq}.csv')
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['stock'] = stock_rank  # 使用权重排名作为股票编号
                required_cols = ['date', 'stock', 'open', 'high', 'low', 'close', 'volume']
                col_mapping = {
                    'Date': 'date',
                    'Open': 'open', 
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }
                df.rename(columns=col_mapping, inplace=True, errors='ignore')
                if all(col in df.columns for col in required_cols):
                    df = df[required_cols]
                    all_data.append(df)
                    print(f"股票 {stock_code} (权重排名第{stock_rank}) -> 股票编号 {stock_rank}")
                else:
                    print(f"警告：文件 {file_path} 缺少必要的列，已跳过")
            else:
                print(f"警告：找不到股票 {stock_code} 的数据文件")

        merged_data = pd.concat(all_data, ignore_index=True)
        merged_data.sort_values(['stock', 'date'], inplace=True)
        merged_data.to_csv('./data/{}_{}_{}.csv'.format(self.market_name, self.topK, self.freq), index=False)

        print(f"已成功将{len(all_data)}个股票数据合并，按权重排序分配股票编号")

    def run(self):
        self.get_constituent_by_weight() # 根据权重获取CSI300前topK支股票列表
        self.get_market_index_data() # Get market index data
        self.get_stock_price_data() # Get stock price data
        self.merge_stock_data() # Merge stock data
        print("Done!")
        # self.demo() # DEMO for getting the data
        pass

def check_account_quota():

    quota_info = rqdatac.user.get_quota()
    print(quota_info)

def main():
    rqextractor = RiceQuantExtractor()
    rqextractor.run()

if __name__ == '__main__':
    main()
