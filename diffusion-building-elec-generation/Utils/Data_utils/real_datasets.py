import os
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask


class CustomDataset(Dataset):
    def __init__(
        self, 
        name,
        data_root, 
        window=64, 
        proportion=0.8, 
        save2npy=True, 
        neg_one_to_one=True,
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        predict_length=None,
        missing_ratio=None,
        style='separate', 
        distribution='geometric', 
        mean_mask_length=3
    ):
        super(CustomDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        self.rawdata, self.scaler = self.read_data(data_root, self.name)
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]
        self.sample_num_total = max(self.len - self.window + 1, 0)
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one

        self.data = self.__normalize(self.rawdata)
        train, inference = self.__getsamples(self.data, proportion, seed)

        self.samples = train if period == 'train' else inference
        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()
        self.sample_num = self.samples.shape[0]

    def __getsamples(self, data, proportion, seed):
        proportion = proportion*10
        x = np.zeros((self.sample_num_total, self.window, self.var_num))
        for i in range(self.sample_num_total):
            start = i
            end = i + self.window
            x[i, :, :] = data[start:end, :]

        train_data, test_data = self.divide(x, proportion, seed)

        #train_data, test_data = self.divide2(x, proportion)
        if self.save2npy:
            if 1 - proportion > 0:
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize(test_data))
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), self.unnormalize(train_data))
            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize_to_zero_to_one(test_data))
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), unnormalize_to_zero_to_one(train_data))
            else:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data)
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data)

        return train_data, test_data

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)
    
    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)
    
    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.shape[0]

        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        if ratio == 0.2:
            # Divide into 2 parts, take the first (odd-indexed) part
            num_splits = 2
            regular_train_num = size // 2
            id_rdm = np.arange(size)
            regular_train_id = id_rdm[:regular_train_num]
            irregular_train_id = id_rdm[regular_train_num:]
        elif ratio == 0.4:
            # Divide into 4 parts, take the 1st, 3rd parts (odd-indexed)
            num_splits = 4
            split_size = size // 4
            id_rdm = np.arange(size)
            regular_train_id = id_rdm[:split_size]  # 1st part
            regular_train_id = np.concatenate([regular_train_id, id_rdm[2 * split_size: 3 * split_size]])  # 3rd part
            irregular_train_id = np.setdiff1d(id_rdm, regular_train_id)  # Remaining parts: 2nd, 4th
        elif ratio == 0.8:
            # Divide into 8 parts, take the 1st, 3rd, 5th, 7th parts (odd-indexed)
            num_splits = 8
            split_size = size // 8
            id_rdm = np.arange(size)
            regular_train_id = id_rdm[:split_size]  # 1st part
            regular_train_id = np.concatenate([regular_train_id, id_rdm[2 * split_size: 3 * split_size],
                                               id_rdm[4 * split_size: 5 * split_size],
                                               id_rdm[6 * split_size: 7 * split_size]])  # 3rd, 5th, 7th parts
            irregular_train_id = np.setdiff1d(id_rdm, regular_train_id)  # Remaining parts: 2nd, 4th, 6th, 8th
        else:
            # For ratio <= 1, we simply split as before
            regular_train_num = int(np.ceil(size * ratio))
            id_rdm = np.arange(size)
            regular_train_id = id_rdm[:regular_train_num]
            irregular_train_id = id_rdm[regular_train_num:]

        # Split data based on the calculated indices
        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Restore RNG
        np.random.set_state(st0)

        return regular_data, irregular_data

    @staticmethod
    def divide2(self, x, proportion):
        """
        按时间顺序划分数据（无随机性）

        参数：
            x: 时间序列数据 (N, ...)
            proportion: 训练集使用的“月份数”（例如 3、6、9、12）

        返回：
            train_data, test_data
        """

        # 每个月的数据量：24小时 × 30天
        month_len = 24 * 30  # 720

        # 训练数据长度
        train_len = int(month_len * proportion)

        # 边界保护
        if train_len > len(x):
            raise ValueError(f"训练数据长度 {train_len} 超过总数据长度 {len(x)}")

        # 按时间顺序切分
        train_data = x[:train_len]
        test_data = x[train_len:]

        return train_data, test_data


    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        df = pd.read_csv(filepath, header=0)
        if name == 'etth':
            df.drop(df.columns[0], axis=1, inplace=True)
        data = df.values
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler
    
    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if self.period == 'test':
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num


class CustomDatasetOURS(Dataset):
    def __init__(
            self,
            name,
            data_root,
            window=64,
            proportion=0.8,
            save2npy=True,
            neg_one_to_one=True,
            seed=123,
            period='train',
            output_dir='./OUTPUT',
            predict_length=None,
            missing_ratio=None,
            style='separate',
            distribution='geometric',
            mean_mask_length=3
    ):
        super(CustomDatasetOURS, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        self.rawdata, self.scaler = self.read_data(data_root, self.name)
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]
        self.sample_num_total = max(self.len - self.window + 1, 0)
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one

        self.data = self.__normalize(self.rawdata)
        train, inference = self.__getsamples(self.data, proportion, seed)

        self.samples = train if period == 'train' else inference
        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()
        self.sample_num = self.samples.shape[0]

    def __getsamples(self, data, proportion, seed):
        x = np.zeros((self.sample_num_total, self.window, self.var_num))
        for i in range(self.sample_num_total):
            start = i
            end = i + self.window
            x[i, :, :] = data[start:end, :]

        train_data, test_data = self.divide2(x, proportion)

        if self.save2npy:

            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"),
                        self.unnormalize(test_data))
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"),
                    self.unnormalize(train_data))
            if self.auto_norm:

                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"),
                            unnormalize_to_zero_to_one(test_data))
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"),
                        unnormalize_to_zero_to_one(train_data))
            else:

                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data)
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data)

        return train_data, test_data

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)

    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)

    # @staticmethod
    # def divide(data, ratio, seed=2023):
    #     size = data.shape[0]
    #
    #     # Store the state of the RNG to restore later.
    #     st0 = np.random.get_state()
    #     np.random.seed(seed)
    #
    #     if ratio == 0.2:
    #         # Divide into 2 parts, take the first (odd-indexed) part
    #         num_splits = 2
    #         regular_train_num = size // 2
    #         id_rdm = np.arange(size)
    #         regular_train_id = id_rdm[:regular_train_num]
    #         irregular_train_id = id_rdm[regular_train_num:]
    #     elif ratio == 0.4:
    #         # Divide into 4 parts, take the 1st, 3rd parts (odd-indexed)
    #         num_splits = 4
    #         split_size = size // 4
    #         id_rdm = np.arange(size)
    #         regular_train_id = id_rdm[:split_size]  # 1st part
    #         regular_train_id = np.concatenate([regular_train_id, id_rdm[2 * split_size: 3 * split_size]])  # 3rd part
    #         irregular_train_id = np.setdiff1d(id_rdm, regular_train_id)  # Remaining parts: 2nd, 4th
    #     elif ratio == 0.8:
    #         # Divide into 8 parts, take the 1st, 3rd, 5th, 7th parts (odd-indexed)
    #         num_splits = 8
    #         split_size = size // 8
    #         id_rdm = np.arange(size)
    #         regular_train_id = id_rdm[:split_size]  # 1st part
    #         regular_train_id = np.concatenate([regular_train_id, id_rdm[2 * split_size: 4 * split_size],
    #                                            id_rdm[4 * split_size: 5 * split_size],
    #                                            id_rdm[6 * split_size: 8 * split_size]])  # 3rd, 5th, 7th parts
    #         irregular_train_id = np.setdiff1d(id_rdm, regular_train_id)  # Remaining parts: 2nd, 4th, 6th, 8th
    #     else:
    #         # For ratio <= 1, we simply split as before
    #         regular_train_num = int(np.ceil(size * (ratio+0.4*(1-ratio))))
    #         id_rdm = np.arange(size)
    #         regular_train_id = id_rdm[:regular_train_num]
    #         irregular_train_id = id_rdm[regular_train_num:]
    #
    #     # Split data based on the calculated indices
    #     regular_data = data[regular_train_id, :]
    #     irregular_data = data[irregular_train_id, :]
    #
    #     # Restore RNG
    #     np.random.set_state(st0)
    #
    #     return regular_data, irregular_data
    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.shape[0]

        # 1. 映射比例 (Mapping Logic)
        # 根据你的需求：0.9->0.9, 0.7->0.8, 0.5->0.65, 0.3->0.5
        # 以及：0.8->0.6随机, 0.4->0.55随机, 0.2->0.5随机
        if ratio == 0.9:
            target_ratio = 0.9
        elif ratio == 0.8:
            target_ratio = 0.6
        elif ratio == 0.7:
            target_ratio = 0.8
        elif ratio == 0.5:
            target_ratio = 0.65
        elif ratio == 0.4:
            target_ratio = 0.55
        elif ratio == 0.3:
            target_ratio = 0.5
        elif ratio == 0.2:
            target_ratio = 0.5
        else:
            # 默认处理其他比例
            target_ratio = ratio

        # 2. 执行随机抽取
        st0 = np.random.get_state()
        np.random.seed(seed)

        # 生成打乱的索引
        indices = np.arange(size)
        np.random.shuffle(indices)

        # 计算训练集数量
        train_num = int(size * target_ratio)

        # 分配索引
        regular_train_id = indices[:train_num]
        irregular_train_id = indices[train_num:]

        # 恢复 RNG 状态
        np.random.set_state(st0)

        # 3. 提取数据
        # 注意：为了保持时间序列的相对顺序（可选），可以对索引进行排序
        # regular_train_id.sort()
        # irregular_train_id.sort()

        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        return regular_data, irregular_data

    @staticmethod
    def divide2(x, proportion):
        """
        按时间顺序划分数据（无随机性）

        参数：
            x: 时间序列数据 (N, ...)
            proportion: 训练集使用的“月份数”（例如 3、6、9、12）

        返回：
            train_data, test_data
        """

        # 每个月的数据量：24小时 × 30天
        month_len = 24 * 30  # 720

        # 训练数据长度
        train_len = int(month_len * proportion)

        # 边界保护
        # if train_len > len(x):
        #     raise ValueError(f"训练数据长度 {train_len} 超过总数据长度 {len(x)}")

        # 按时间顺序切分
        print(type(x))
        train_data = x[:train_len]
        test_data = x[train_len:]

        return train_data, test_data
    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        df = pd.read_csv(filepath, header=0)
        if name == 'etth':
            df.drop(df.columns[0], axis=1, inplace=True)
        data = df.values
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler

    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if self.period == 'test':
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num
class fMRIDataset(CustomDataset):
    def __init__(
        self, 
        proportion=1., 
        **kwargs
    ):
        super().__init__(proportion=proportion, **kwargs)

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        data = io.loadmat(filepath + '/sim4.mat')['ts']
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler
