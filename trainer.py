import os
import torch
from tqdm import tqdm
from transformers import WhisperModel, Wav2Vec2Model
from torch.utils.tensorboard import SummaryWriter
from lion_pytorch import Lion


def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data

class Trainer:
    def __init__(self, args, model, dataset):
        self.args = args
        self.model = model
        self.get_optimizer()
        self.train_loader = cycle(dataset['train'])
        self.valid_loader = cycle(dataset['valid'])
        self.min_valid_loss = 999999
        self.loss_log = []
        self.writer = SummaryWriter(log_dir=args.run_dir)
        if args.audio_encoder == 'whisper':
            self.audio_encoder = WhisperModel.from_pretrained("openai/whisper-small").encoder.cuda()
            self.audio_encoder._freeze_parameters()
        elif args.audio_encoder == 'wav2vec':
            self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").cuda()
            self.audio_encoder.freeze_feature_encoder()

    def get_optimizer(self):
        if self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr)
        elif self.args.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr)
        elif self.args.optimizer == 'lion':
            self.optimizer = Lion(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr)

    def save(self, model_name):
        torch.save(self.model.state_dict(), os.path.join(self.args.model_dir, model_name))

    def train(self):
        for iters in (pbar := tqdm(range(1, self.args.max_iter + 1))):
            # training
            loss = self.training_step()
            self.loss_log.append(loss)
            loss_window = self.loss_log[-self.args.loss_wid:]
            loss_avg = sum(loss_window) / len(loss_window)
            pbar.set_description(f"TRAIN LOSS:{loss_avg:.7f}")
            if iters % self.args.log_interval == 0:
                self.writer.add_scalar('Loss/train', loss, iters)

            # validating
            if iters % self.args.valid_interval == 0:
                valid_loss = []
                for valid_iter in range(self.args.valid_steps):
                    loss = self.validation_step()
                    valid_loss.append(loss)
                valid_loss_avg = sum(valid_loss) / len(valid_loss)
                self.writer.add_scalar('Loss/valid', valid_loss_avg, iters)
                print(f"Iter: {iters} Valid Loss: {valid_loss_avg:.7f}")
                if valid_loss_avg < self.min_valid_loss:
                    self.min_valid_loss = valid_loss_avg
                self.save(model_name=f"iter{iters}_valid{valid_loss_avg:.7f}.model")

    def training_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        audio, displace = next(self.train_loader)
        audio, displace = audio.to(device="cuda"), displace.to(device="cuda")
        with torch.no_grad():
            audio_embedding = self.audio_encoder(audio).last_hidden_state
        loss = self.model(audio_embedding, displace)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validation_step(self):
        self.model.eval()
        audio, displace = next(self.valid_loader)
        audio, displace = audio.to(device="cuda"), displace.to(device="cuda")
        with torch.no_grad():
            audio_embedding = self.audio_encoder(audio).last_hidden_state
            loss = self.model(audio_embedding, displace)
        return loss.item()

