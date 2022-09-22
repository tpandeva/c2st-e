import time
import numpy as np
import torch
from torch import nn
import torch.fft
from collections import defaultdict

from .unet import Unet


class ModelModule:
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
        do_early_stopping=True,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = PathologyClassifier(
            in_chans, chans, num_pool_layers, drop_prob, input_shape
        ).to(self.device)

        self.early_stopping = do_early_stopping

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

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.lr_gamma = total_lr_gamma ** (1 / num_epochs)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimiser, self.lr_gamma
        )

    def train_epoch(self, loader):
        start_time = time.perf_counter()

        self.model.train()
        self.bce_loss.train()

        extra_output = {}
        epoch_loss = 0
        total_samples = 0
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
            ) = sample
            total_samples += image.shape[0]

            self.optimiser.zero_grad()

            image = image.unsqueeze(1).to(self.device)
            # Positive class = any pathology
            target = (
                (slice_pathologies.sum(dim=1) > 0).unsqueeze(1).float().to(self.device)
            )

            logits = self.model(image)
            loss = self.bce_loss(logits, target)
            reduced_loss = loss.mean(dim=1).sum(dim=0)  # Sum over batch dim

            epoch_loss += reduced_loss.item()

            reduced_loss.backward()
            self.optimiser.step()

        self.scheduler.step()

        epoch_loss /= total_samples
        extra_output["train_epoch_time"] = time.perf_counter() - start_time
        return epoch_loss, extra_output

    def val_epoch(self, loader):
        start_time = time.perf_counter()

        self.model.eval()
        self.bce_loss.eval()

        with torch.no_grad():
            extra_output = {}
            epoch_loss = 0
            epoch_acc = 0
            total_samples = 0
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
                ) = sample
                total_samples += image.shape[0]

                # Debugging: visualise images
                #                 plt.imshow(image.cpu().numpy()[0])
                #                 plt.show()
                #                 break

                image = image.unsqueeze(1).to(self.device)
                # Positive class = any pathology
                target = (
                    (slice_pathologies.sum(dim=1) > 0)
                    .unsqueeze(1)
                    .float()
                    .to(self.device)
                )
                # target = torch.stack((target, 1-target), dim=1).to(self.device)

                logits = self.model(image)

                # Accuracy
                labels = torch.sigmoid(logits) > 0.5
                epoch_acc += (labels == target.byte()).sum().float().item()
                # Loss
                loss = self.bce_loss(logits, target)
                reduced_loss = loss.mean(dim=1).sum(dim=0)  # Sum over batch dim

                epoch_loss += reduced_loss.item()

            epoch_loss /= total_samples
            epoch_acc /= total_samples

        extra_output["val_epoch_time"] = time.perf_counter() - start_time
        return epoch_loss, epoch_acc, extra_output

    def train(self, train_loader, val_loader=None, print_every=10, eval_every=10):
        start_time = time.perf_counter()
        extra_output = defaultdict(lambda: defaultdict(dict))

        train_losses = []
        val_losses = {}  # Not computed every epoch, so dict to keep track of epochs.
        val_accs = {}
        best_val_loss = 1000
        for epoch in range(self.num_epochs):
            if epoch % print_every == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}")
            if val_loader is not None and epoch % eval_every == 0:
                val_loss, val_acc, val_extra_output = self.val_epoch(val_loader)
                if self.early_stopping:
                    if val_loss <= best_val_loss:
                        best_val_loss = val_loss
                    else:  # Stop training at this epoch: should technically have stopped at model of previous epoch.
                        print("Stopping early...")
                        break
                val_losses[epoch] = val_loss
                val_accs[epoch] = val_acc
                extra_output[epoch]["val"] = val_extra_output
                print(
                    f"   Val loss: {val_loss:.3f}, Val acc: {val_acc:.2f}, time: {val_extra_output['val_epoch_time']:.2f}s"
                )

            train_loss, train_extra_output = self.train_epoch(train_loader)
            print(
                f" Train loss: {train_loss:.3f}, time: {train_extra_output['train_epoch_time']:.2f}s"
            )
            train_losses.append(np.mean(train_loss))
            extra_output[epoch]["train"] = train_extra_output

        val_loss, val_acc, val_extra_output = self.val_epoch(val_loader)
        val_losses[epoch] = val_loss
        val_accs[epoch] = val_acc
        extra_output[epoch]["val"] = val_extra_output
        print(
            f"   Val loss: {val_loss:.3f}, Val acc: {val_acc:.2f}, time: {val_extra_output['val_epoch_time']:.2f}s"
        )

        return (
            train_losses,
            val_losses,
            val_accs,
            extra_output,
            time.perf_counter() - start_time,
        )

    def test(self, loader):
        start_time = time.perf_counter()
        extra_output = {}
        all_logits = []
        all_targets = []

        self.model.eval()
        self.bce_loss.eval()
        with torch.no_grad():
            test_loss = 0
            test_acc = 0
            total_samples = 0
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
                ) = sample
                total_samples += image.shape[0]
                # Preprocessing
                image = image.unsqueeze(1).to(self.device)
                # Positive class = any pathology
                target = (slice_pathologies.sum(dim=1) > 0).unsqueeze(1).float()
                all_targets.append(target)
                target = target.to(self.device)
                # target = torch.stack((pathology_yes_no, 1-pathology_yes_no), dim=1).to(self.device)

                # Run model
                logits = self.model(image)
                all_logits.append(logits.cpu())

                # Accuracy
                labels = torch.sigmoid(logits) > 0.5
                test_acc += (labels == target.byte()).sum().float().item()

                # Loss
                loss = self.bce_loss(logits, target)
                reduced_loss = loss.mean(dim=1).sum(dim=0)  # Sum over batch dim
                test_loss += reduced_loss.item()

            test_loss /= total_samples
            test_acc /= total_samples

        test_time = time.perf_counter() - start_time
        extra_output["test_time"] = test_time
        extra_output["logits"] = torch.cat(all_logits, axis=0)
        extra_output["targets"] = torch.cat(all_targets, axis=0)
        print(
            f" Test loss: {test_loss:.3f}, Test acc: {test_acc:.2f}, time: {test_time:.2f}s"
        )
        return test_loss, test_acc, extra_output


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


class PathologyClassifier(nn.Module):
    def __init__(self, in_chans, chans, num_pool_layers, drop_prob, input_shape):
        super().__init__()

        assert (
            input_shape[0] == input_shape[1]
        ), "`enc_size` computation assumes square images (potentially)."
        up_factor = chans * 2 ** num_pool_layers
        down_factor = 4 ** num_pool_layers
        self.enc_size = input_shape[0] * input_shape[1] * up_factor // down_factor

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=1,  # Unused
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob,
        )
        self.linear = LinearModel(self.enc_size, 1)

    def forward(self, image):
        enc, _ = self.unet.encoder(
            image
        )  # Shape: [N, C, H, W] = [16, 256, 20, 20] (e.g.)
        enc = torch.flatten(
            enc, start_dim=1
        )  # Shape: [N, enc_size] = [16, 102400] (e.g.)
        logits = self.linear(enc)
        return logits
