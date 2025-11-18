from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.rec_criterion = nn.MSELoss()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.data == 'PEMS':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # channel_decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # fc1 - channel_decoder
                if self.args.output_attention:
                    outputs,x,other_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,is_train=False)[0]
                else:
                    outputs,x,other_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,is_train=False)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                self.cpu = batch_y.detach().cpu()
                true = self.cpu

                if self.args.data == 'PEMS':
                    B, T, C = pred.shape
                    pred = pred.cpu().numpy()
                    true = true.cpu().numpy()
                    pred = vali_data.inverse_transform(pred.reshape(-1, C)).reshape(B, T, C)
                    true = vali_data.inverse_transform(true.reshape(-1, C)).reshape(B, T, C)
                    mae, mse, rmse, mape, mspe = metric(pred, true)
                    loss = mae
                else:
                    loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.pre_epoches):
            if epoch == 0:  # 仅在预训练的第一个 epoch 初始化列表
                batch_x_list = []
                batch_y_list = []
                batch_x_mark_list = []
                batch_y_mark_list = []
                batch_u_list = []
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # 使用 TimeBridge 签名调用模型，但设置 is_out_u=True
                outputs, x_rec, other_loss, U = self.model(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark,
                    y_enc=batch_y, is_train=True, is_out_u=True, c_est=None
                )

                # 收集用于阶段二的数据
                # (仅在最后一个预训练 epoch 或 early stopping 时收集)
                if epoch == self.args.pre_epoches - 1 or early_stopping.counter == self.args.patience - 1:
                    batch_x_list.append(batch_x.cpu())
                    batch_y_list.append(batch_y.cpu())
                    batch_x_mark_list.append(batch_x_mark.cpu())
                    batch_y_mark_list.append(batch_y_mark.cpu())
                    batch_u_list.append(U.detach().cpu())

                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                # 计算 Time-Freq MAE 损失
                loss = self.time_freq_mae(batch_y_target, outputs)

                # 添加来自 VAE/HMM 的损失 (other_loss)
                loss += other_loss
                # 添加 TimeBridge 的重建损失
                rec_loss = self.time_freq_mae(batch_x, x_rec) * self.args.rec_weight
                loss += rec_loss

                # print(f"Epoch {epoch}, Batch {i}:")
                # print(f"  Time-Freq MAE Loss: {loss1.item():.6f}")
                # print(f"  Other Loss (KL+HMM): {other_loss.item():.6f}")
                # print(f"  rec Loss: {rec_loss.item():.6f}")
                # print(f"  Total Loss: {loss.item():.6f}")

                train_loss.append(loss.item())

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 添加梯度裁剪以提高HMM稳定性
                model_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping during pre-training")
                break
            adjust_learning_rate(model_optim, None, epoch + 1, self.args)

        # 阶段二：训练 VAE/Transformer (来自 exp_nsts_with_pre.py)

        # 检查是否收集到了数据

            # 用收集到的 HMM 状态 (U) 创建新的数据集和加载器
        dataset = TensorDataset(
            torch.cat(batch_u_list, dim=0),
            torch.cat(batch_x_list, dim=0),
            torch.cat(batch_y_list, dim=0),
            torch.cat(batch_x_mark_list, dim=0),
            torch.cat(batch_y_mark_list, dim=0)
        )
        train_loader = DataLoader(dataset, self.args.batch_size, shuffle=True)

        # 重置 early stopping
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        for epoch in range(self.args.pre_epoches, self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            # 注意：新的 train_loader 包含 batch_u
            for i, (batch_u, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                # 将所有数据移动到 device
                batch_u = batch_u.to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # TimeBridge 需要的额外输入
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # 使用 TimeBridge 签名调用模型，但这次传入 c_est=batch_u
                # HMM 将被冻结，只训练 VAE/Transformer
                outputs, x_rec, other_loss, _ = self.model(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark,
                    y_enc=batch_y, is_train=True, is_out_u=True, c_est=batch_u
                )

                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                loss = self.time_freq_mae(batch_y_target, outputs)
                loss += other_loss  # 添加 VAE KL 损失

                # (可选) 添加 TimeBridge 的重建损失
                loss += self.time_freq_mae(batch_x, x_rec) * self.args.rec_weight

                train_loss.append(loss.item())

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                model_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, None, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def time_freq_mae(self, batch_y, outputs):
        # time mae loss
        t_loss = (outputs - batch_y).abs().mean()

        # freq mae loss
        f_loss = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)
        f_loss = f_loss.abs().mean()

        return (1 - self.args.alpha) * t_loss + self.args.alpha * f_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # channel_decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # fc1 - channel_decoder
                if self.args.output_attention:
                    outputs,x,other_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,y_enc=None,is_train=False, is_out_u=False, c_est=None)[0]

                else:
                    outputs,x,other_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,y_enc=None,is_train=False, is_out_u=False, c_est=None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    outputs = test_data.inverse_transform(outputs)
                    batch_y = test_data.inverse_transform(batch_y)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        if self.args.data == 'PEMS':
            B, T, C = preds.shape
            preds = test_data.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
            trues = test_data.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        print('rmse:{}, mape:{}, mspe:{}'.format(rmse, mape, mspe))
        f = open(f"result_long_term_forecast_{self.args.data_path}_.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()

        return
