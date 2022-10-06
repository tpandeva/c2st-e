import time
import pickle
import numpy as np
import torch
from torch import nn
import torch.fft
from collections import defaultdict
from .unet import Unet

import sys
sys.path.append("..")
from tests import MMDu  # noqa


class SamplingModelModuleMMD:
    # Automatically does Type-Ia, Ib, II integration.
    def __init__(
        self,
        in_chans,
        chans,
        num_pool_layers,
        drop_prob,
        input_shape,
        lr,
        total_lr_gamma,
        num_epochs,
        save_dir,
        do_early_stopping=True,
        patience=3,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = PathologyClassifierMMD(in_chans, chans, num_pool_layers, drop_prob, input_shape).to(self.device)

        self.embedding_save_dir = save_dir / "embeddings"
        self.embedding_save_dir.mkdir(parents=True, exist_ok=False)

        self.early_stopping = do_early_stopping
        self.patience = patience

        # Architecture params
        self.in_chans = in_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.input_shape = input_shape

        # Optimiser params
        self.lr = lr
        self.total_lr_gamma = total_lr_gamma
        self.num_epochs = num_epochs

        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.lr_gamma = total_lr_gamma ** (1 / num_epochs)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimiser, self.lr_gamma)

    def train_epoch(self, loader):
        start_time = time.perf_counter()
        self.model.train()
        extra_output = {}
        # Run loader to get all positive and negative examples
        all_inputs_neg, all_inputs_pos = [], []
        for i, sample in enumerate(loader):
            (
                kspace,
                image,
                mean,
                std,
                attrs,
                fname,
                dataslice,
                slice_pathologies,
                label,
            ) = sample

            all_inputs_neg.append(image[label == 0])
            all_inputs_pos.append(image[label == 1])

        neg_tensor = torch.cat(all_inputs_neg, dim=0).unsqueeze(1).to(self.device)
        pos_tensor = torch.cat(all_inputs_pos, dim=0).unsqueeze(1).to(self.device)

        # Kxyxy depends on sample size, but it is only used for the final statistical test, so we can do batch training.
        # batch_size = loader.batch_size
        full_size = neg_tensor.shape[0] + pos_tensor.shape[0]
        batch_size = loader.batch_size
        assert batch_size % 2 == 0, "Must use even size batches."
        num_batches = (
            full_size // batch_size
            if full_size % batch_size == 0
            else full_size // batch_size + 1
        )
        epoch_loss = 0
        for k in range(num_batches):
            # Half of each
            neg = neg_tensor[k * batch_size // 2: (k + 1) * batch_size // 2]
            pos = pos_tensor[k * batch_size // 2: (k + 1) * batch_size // 2]
            mmd2, varEst, Kxyxy = self.model(neg, pos)
            mmd_value_temp = -1 * mmd2
            mmd_std_temp = torch.sqrt(varEst + 10 ** (-8))
            loss = torch.div(mmd_value_temp, mmd_std_temp)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            epoch_loss += loss.detach() * neg.shape[0] * 2

        epoch_loss /= full_size
        self.scheduler.step()

        extra_output["train_epoch_time"] = time.perf_counter() - start_time
        return epoch_loss, extra_output

    def val_epoch(self, loader):
        start_time = time.perf_counter()
        self.model.eval()
        with torch.no_grad():
            extra_output = {}
            # Run loader to get all positive and negative examples
            all_inputs_neg, all_inputs_pos = [], []

            for i, sample in enumerate(loader):
                (
                    kspace,
                    image,
                    mean,
                    std,
                    attrs,
                    fname,
                    dataslice,
                    slice_pathologies,
                    label,
                ) = sample
                all_inputs_neg.append(image[label == 0])
                all_inputs_pos.append(image[label == 1])

            neg_tensor = torch.cat(all_inputs_neg, dim=0).unsqueeze(1).to(self.device)
            pos_tensor = torch.cat(all_inputs_pos, dim=0).unsqueeze(1).to(self.device)

            # Kxyxy depends on sample size, but it is only used for the final statistical test, so we can
            # do batch validation.
            full_size = neg_tensor.shape[0] + pos_tensor.shape[0]
            batch_size = loader.batch_size
            assert batch_size % 2 == 0, "Must use even size batches."
            num_batches = (
                full_size // batch_size
                if full_size % batch_size == 0
                else full_size // batch_size + 1
            )
            epoch_loss = 0
            for k in range(num_batches):
                # Half of each
                neg = neg_tensor[k * batch_size // 2: (k + 1) * batch_size // 2]
                pos = pos_tensor[k * batch_size // 2: (k + 1) * batch_size // 2]
                mmd2, varEst, Kxyxy = self.model(neg, pos)
                mmd_value_temp = -1 * mmd2
                mmd_std_temp = torch.sqrt(varEst + 10 ** (-8))
                epoch_loss += torch.div(mmd_value_temp, mmd_std_temp) * neg.shape[0] * 2
            epoch_loss /= full_size

        extra_output["val_epoch_time"] = time.perf_counter() - start_time
        return epoch_loss, extra_output

    def train(self, train_loader, val_loader=None, print_every=10, eval_every=10):
        start_time = time.perf_counter()
        extra_output = defaultdict(lambda: defaultdict(dict))

        train_losses = []
        val_losses = {}  # Not computed every epoch, so dict to keep track of epochs.
        best_val_loss = 1000
        patience_count = 0
        for epoch in range(self.num_epochs):
            if epoch % print_every == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}")
            if val_loader is not None and epoch % eval_every == 0:
                val_loss, val_extra_output = self.val_epoch(val_loader)
                # Early stopping block
                if self.early_stopping:
                    if val_loss <= best_val_loss:
                        patience_count = 0
                        best_val_loss = val_loss
                    else:
                        patience_count += 1

                    # Stop training at this epoch: should technically have stopped before model performance dropped.
                    if patience_count == self.patience:
                        print("Stopping early...")
                        break
                val_losses[epoch] = val_loss
                extra_output[epoch]["val"] = val_extra_output
                print(
                    f"   Val loss: {val_loss:.3f}, time: {val_extra_output['val_epoch_time']:.2f}s"
                )

            train_loss, train_extra_output = self.train_epoch(train_loader)
            print(f" Train loss: {train_loss:.3f}, time: {train_extra_output['train_epoch_time']:.2f}s")
            train_losses.append(train_loss)
            extra_output[epoch]["train"] = train_extra_output

        val_loss, val_extra_output = self.val_epoch(val_loader)
        val_losses[epoch] = val_loss
        extra_output[epoch]["val"] = val_extra_output
        print(f"   Val loss: {val_loss:.3f}, time: {val_extra_output['val_epoch_time']:.2f}s")

        return (
            train_losses,
            val_losses,
            extra_output,
            time.perf_counter() - start_time,
        )

    def test(self, loader):
        start_time = time.perf_counter()
        extra_output = {}
        self.model.eval()
        with torch.no_grad():
            all_inputs_neg, all_inputs_pos = [], []
            for i, sample in enumerate(loader):
                (
                    kspace,
                    image,
                    mean,
                    std,
                    attrs,
                    fname,
                    dataslice,
                    slice_pathologies,
                    label,
                ) = sample
                all_inputs_neg.append(image[label == 0])
                all_inputs_pos.append(image[label == 1])

            neg_tensor = torch.cat(all_inputs_neg, dim=0).unsqueeze(1).to(self.device)
            pos_tensor = torch.cat(all_inputs_pos, dim=0).unsqueeze(1).to(self.device)

            # Cannot run all slices through the model at the same time, so do neg first and pos next, save to disk
            #  in between. Then load and run through MMDu (see forward). This is necessary because Kxyxy depends on
            #  sample size.
            ordered_tensor = torch.cat((neg_tensor, pos_tensor), dim=0)
            batch_size = loader.batch_size
            num_batches = (
                ordered_tensor.shape[0] // batch_size
                if ordered_tensor.shape[0] % batch_size == 0
                else ordered_tensor.shape[0] // batch_size + 1
            )
            print(self.embedding_save_dir)
            for k in range(num_batches):
                batch = ordered_tensor[k * batch_size: (k + 1) * batch_size]
                batch_xy = batch
                batch_xy_hat = self.model.linear(
                    torch.flatten(self.model.unet.encoder(batch_xy)[0], start_dim=1)
                )
                emb_filename = f"{k}_batch.pkl"
                with open(self.embedding_save_dir / emb_filename, "wb") as f:
                    pickle.dump(batch_xy_hat, f)

            x, y = neg_tensor, pos_tensor
            # Load embeddings in correct order (so that they match x and y)
            # TODO: Check
            xy_hat = []
            for emb_file in sorted(self.embedding_save_dir.iterdir(), key=lambda x: int(x.name.split("_")[0])):
                print(emb_file.name, int(emb_file.name.split("_")[0]))
                with open(emb_file, "rb") as f:
                    emb = pickle.load(f)
                    xy_hat.append(emb)
            xy_hat = torch.cat(xy_hat, dim=0)

            mmd2, varEst, Kxyxy = MMDu(
                xy_hat[0: x.shape[0], :],
                xy_hat[x.shape[0]:, :],
                x.view(x.shape[0], -1),
                y.view(y.shape[0], -1),
                self.model.sigma ** 2,
                self.model.sigma0_u ** 2,
                torch.exp(self.model.eps) / (1 + torch.exp(self.model.eps)),
            )

            mmd_value_temp = -1 * mmd2
            mmd_std_temp = torch.sqrt(varEst + 10 ** (-8))
            test_loss = torch.div(mmd_value_temp, mmd_std_temp)

            mmd_size = neg_tensor.shape[0]
            assert mmd_size == pos_tensor.shape[0], "Must have same number of positive and negative examples!"

        test_time = time.perf_counter() - start_time
        extra_output["test_time"] = test_time

        print(f" Test loss: {test_loss:.3f}, time: {test_time:.2f}s")
        return test_loss.item(), Kxyxy, mmd_size, extra_output


class LinearModel(nn.Module):
    def __init__(self, enc_size, output_size):
        super().__init__()

        self.enc_size = enc_size
        self.output_size = output_size

        hidden_size = 512
        self.linear = nn.Linear(enc_size, output_size)
        self.linear1 = nn.Linear(enc_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, encoding):
        #         x = F.relu(self.linear1(encoding))
        #         x = self.linear2(x)
        #         return x
        return self.linear(encoding)


class PathologyClassifierMMD(nn.Module):
    def __init__(self, in_chans, chans, num_pool_layers, drop_prob, input_shape, mmd_enc_size=100):
        super().__init__()

        assert input_shape[0] == input_shape[1], "`enc_size` computation assumes square images (potentially)."
        up_factor = chans * 2 ** num_pool_layers
        down_factor = 4 ** num_pool_layers
        self.enc_size = input_shape[0] * input_shape[1] * up_factor // down_factor
        self.mmd_enc_size = mmd_enc_size

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=1,  # Unused
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob,
        )
        self.linear = nn.Linear(self.enc_size, self.mmd_enc_size)

        # MMD params
        self.eps, self.sigma, self.sigma0_u = (
            torch.nn.Parameter(torch.from_numpy(np.random.rand(1) * (10 ** (-10))), requires_grad=True),
            torch.nn.Parameter(torch.from_numpy(np.sqrt(np.random.rand(1) * 0.3)), requires_grad=True),
            torch.nn.Parameter(torch.from_numpy(np.sqrt(np.random.rand(1) * 0.002)), requires_grad=True),
        )

    # def forward(self, image):
    #     enc, _ = self.unet.encoder(image)  # Shape: [N, C, H, W] = [16, 256, 20, 20] (e.g.)
    #     enc = torch.flatten(enc, start_dim=1)  # Shape: [N, enc_size] = [16, 102400] (e.g.)
    #     logits = self.linear(enc)
    #     return logits

    def forward(self, x, y):
        xy = torch.cat((x, y))
        xy_hat = self.linear(
            torch.flatten(self.unet.encoder(xy)[0], start_dim=1)
        )
        # TODO: For test, I need to run this full batch because Kxyxy depends on sample size
        mmd2, varEst, Kxyxy = MMDu(
            xy_hat[0 : x.shape[0], :],
            xy_hat[x.shape[0] :, :],
            x.view(x.shape[0], -1),
            y.view(y.shape[0], -1),
            self.sigma ** 2,
            self.sigma0_u ** 2,
            torch.exp(self.eps) / (1 + torch.exp(self.eps)),
        )
        return mmd2, varEst, Kxyxy