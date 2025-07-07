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

        self.start_date = '2013-08-01'
        self.end_date = '2023-10-01' 
        self.market_name = 'CSI300'
        self.market_code = '000300.XSHG'
        self.freq = '60m' # '1d', '60m', 'tick'
        self.topK = 10
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
            # 修正文件路径，与get_price_data方法中保存的路径一致
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
    
    def get_constituent_weight(self):
        # Collect the weights of constituent stocks in CSI300
        cap_weight = rqdatac.index_weights(order_book_id=self.market_code, date=self.start_date)
        cap_weight = pd.DataFrame(cap_weight)
        cap_weight.reset_index(drop=False, inplace=True)
        cap_weight.rename(columns={'order_book_id': 'code', 0: 'weight'}, inplace=True)
        cap_weight.sort_values(by=['weight'], ascending=False, inplace=True, ignore_index=True)
        cap_weight['rank'] = np.arange(1, len(cap_weight)+1)
        cap_weight = cap_weight.head(self.topK)
        
        print(cap_weight)

        fdirx = os.path.join(self.fdir, 'Weights')  
        os.makedirs(fdirx, exist_ok=True)
        fpath = os.path.join(fdirx, '{}_Weights_{}.csv'.format(self.market_name, self.freq))
        cap_weight.to_csv(fpath, index=False)

    def demo(self):
        mkt_index = rqdatac.get_price(order_book_ids=['000300.XSHG'], start_date='2024-01-01', end_date='2024-10-31', frequency='1d', fields=self.request_fields, skip_suspended=False, market='cn')
        print(mkt_index)

    def _check_stock_data_completeness(self, all_data, weights):
        """
        检查每支股票数据的完整性，统计行数并发现异常
        """
        print("\n" + "="*50)
        print("股票数据完整性检查")
        print("="*50)
        
        # 统计每支股票的数据行数
        stock_counts = {}
        stock_info = {}
        
        for df in all_data:
            if not df.empty:
                stock_id = df['stock'].iloc[0]
                row_count = len(df)
                stock_counts[stock_id] = row_count
                
                # 获取股票代码用于显示
                stock_code = weights[weights['rank'] == stock_id]['code'].iloc[0] if len(weights[weights['rank'] == stock_id]) > 0 else 'Unknown'
                stock_info[stock_id] = stock_code
        
        if not stock_counts:
            print("警告：没有找到任何股票数据")
            return
            
        # 计算统计信息
        counts_list = list(stock_counts.values())
        avg_count = np.mean(counts_list)
        median_count = np.median(counts_list)
        max_count = max(counts_list)
        min_count = min(counts_list)
        
        print(f"数据行数统计：")
        print(f"  最大行数: {max_count}")
        print(f"  最小行数: {min_count}")
        print(f"  平均行数: {avg_count:.1f}")
        print(f"  中位数行数: {median_count:.1f}")
        print(f"  总共股票数: {len(stock_counts)}")
        
        # 找出最常见的行数（正常情况下的标准行数）
        from collections import Counter
        count_freq = Counter(counts_list)
        most_common_count = count_freq.most_common(1)[0][0]
        most_common_freq = count_freq.most_common(1)[0][1]
        
        print(f"  最常见行数: {most_common_count} (出现{most_common_freq}次)")
        
        # 设置异常检测阈值
        # 如果某股票行数少于最常见行数的90%，则视为异常
        threshold_ratio = 0.9
        threshold_count = most_common_count * threshold_ratio
        
        # 检测异常股票
        abnormal_stocks = []
        for stock_id, count in stock_counts.items():
            if count < threshold_count:
                abnormal_stocks.append((stock_id, count, stock_info[stock_id]))
        
        # 打印详细统计
        print(f"\n各股票数据行数详情：")
        print("-" * 60)
        print(f"{'股票序号':<8} {'股票代码':<15} {'数据行数':<10} {'状态'}")
        print("-" * 60)
        
        for stock_id in sorted(stock_counts.keys()):
            count = stock_counts[stock_id]
            code = stock_info[stock_id]
            status = "正常" if count >= threshold_count else "⚠️ 异常"
            print(f"{stock_id:<8} {code:<15} {count:<10} {status}")
        
        # 打印警告信息
        if abnormal_stocks:
            print(f"\n⚠️  发现 {len(abnormal_stocks)} 支股票数据可能不完整：")
            print("-" * 60)
            for stock_id, count, code in abnormal_stocks:
                missing_ratio = (most_common_count - count) / most_common_count * 100
                print(f"  股票 {stock_id} ({code}): {count} 行数据，比标准少 {missing_ratio:.1f}%")
            print("\n建议检查这些股票的数据下载是否完整！")
        else:
            print(f"\n✅ 所有股票数据行数均正常 (阈值: {threshold_count:.0f} 行)")
        
        print("="*50)

    def merge_stock_data(self):
        fdir = os.path.join(self.fdir, '{}_{}_{}'.format(self.market_name, self.topK, self.freq))
        if not os.path.exists(fdir):
            raise ValueError("Please provide a list of constituent stocks")
            
        # 读取权重文件以获取正确的股票排序
        wpath = os.path.join(self.fdir, 'Weights', '{}_Weights_{}.csv'.format(self.market_name, self.freq))
        if not os.path.exists(wpath):
            raise ValueError("权重文件不存在，无法确定股票序号")
            
        weights = pd.read_csv(wpath)
        weights.sort_values(by=['rank'], ascending=True, inplace=True, ignore_index=True)
        
        all_data = []
        
        # 按权重文件中的rank顺序处理股票数据
        for idx, row in weights.iterrows():
            stock_code = row['code']
            stock_rank = row['rank']  # 这是按权重排序的rank（权重最大的rank=1）
            
            # 构建对应的文件路径
            file_path = os.path.join(fdir, '{}_{}.csv'.format(stock_code, self.freq))
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['stock'] = stock_rank  # 使用权重排序的rank作为股票序号
                
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
                    print(f"已加载股票 {stock_code}，权重排名: {stock_rank}")
                else:
                    print(f"警告：文件 {file_path} 缺少必要的列，已跳过")
            else:
                print(f"警告：股票 {stock_code} 的数据文件不存在，已跳过")

        if not all_data:
            raise ValueError("没有找到任何有效的股票数据文件")
            
        # 统计每支股票的数据行数并检测异常
        self._check_stock_data_completeness(all_data, weights)
            
        merged_data = pd.concat(all_data, ignore_index=True)
        merged_data.sort_values(['stock', 'date'], inplace=True)
        merged_data.to_csv('./data/{}_{}_{}.csv'.format(self.market_name, self.topK, self.freq), index=False)

        print(f"已成功将{len(all_data)}个股票数据合并，按权重排序分配序号（序号1=权重最大）")

    def run(self):
        self.get_constituent_weight() # Get the stock list and capital-based weight of constituent stocks in CSI300
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
