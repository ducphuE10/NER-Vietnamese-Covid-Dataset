import torch
from sklearn.metrics import f1_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer(object):
    def __init__(self, model, data, optimizer, device, loss_fn_cls=None, path='pretrain/model.pt'):
        self.device = device
        self.model = model.to(device)
        self.data = data
        self.optimizer = optimizer(model.parameters())
        self.path = path

    def word_accuracy(self, preds, y):
        assert len(preds) == len(y)
        flatten_preds = [pred for sent_pred in preds for pred in sent_pred]
        flatten_y = [tag for sent_tag in y for tag in sent_tag]
        correct = [pred == tag for pred, tag in zip(flatten_preds, flatten_y)]
        return sum(correct) / len(correct) if len(correct) > 0 else 0

    def f1_score(self, preds, y, full_report=False):
        index_o = self.data.tag_field.vocab.stoi["O"]
        # take all labels except padding and "O"
        positive_labels = [i for i in range(len(self.data.tag_field.vocab.itos))
                           if i not in (self.data.tag_pad_idx, index_o)]

        flatten_pred = [pred for sent_pred in preds for pred in sent_pred]
        flatten_y = [tag for sent_tag in y for tag in sent_tag]
        if full_report:
            # take all names except padding and "O"
            positive_names = [self.data.tag_field.vocab.itos[i]
                              for i in range(len(self.data.tag_field.vocab.itos))
                              if i not in (self.data.tag_pad_idx, index_o)]

            print(classification_report(
                y_true=flatten_y,
                y_pred=flatten_pred,
                labels=positive_labels,
                target_names=positive_names
            ))

        return f1_score(
            y_true=flatten_y,
            y_pred=flatten_pred,
            labels=positive_labels,
            average="micro"
        )

    def sent_accuracy(self, preds, y):
        assert len(preds) == len(y)
        count = 0
        for i in range(len(preds)):
            if preds[i] == y[i]:
                count += 1

        return count / len(preds)

    def train(self, N):
        history = {
            "num_params": self.model.count_parameters(),

            'train_loss': [],
            'val_loss': [],

            'train_f1': [],
            'val_f1': [],

            'train_sent_acc': [],
            'val_sent_acc': [],
        }

        lr_scheduling = ReduceLROnPlateau(
            optimizer=self.optimizer,
            patience=5,
            factor=0.3,
            mode="max",
            verbose=True
        )

        previous_f1 = 0
        for epoch in range(N):
            epoch_loss = 0
            true_tags_epoch = []
            pred_tags_epoch = []
            self.model.train()
            for batch in self.data.train_iter:
                # words = [sent len, batch size]
                words = batch.word.to(self.device)
                # chars = [batch size, sent len, char len]
                chars = batch.char.to(self.device)
                # tags = [sent len, batch size]
                true_tags = batch.tag.to(self.device)

                pred_tags_list, batch_loss = self.model(words, chars, true_tags)
                pred_tags_epoch += pred_tags_list
                # to calculate the loss and f1, we flatten true tags
                true_tags_epoch += [
                    [tag for tag in sent_tag if tag != self.data.tag_pad_idx]
                    for sent_tag in true_tags.permute(1, 0).tolist()
                ]

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                epoch_loss += batch_loss.item()

            epoch_f1 = self.f1_score(pred_tags_epoch, true_tags_epoch, full_report=False)
            epoch_loss = epoch_loss / len(self.data.train_iter)
            epoch_sent_acc = self.sent_accuracy(pred_tags_epoch, true_tags_epoch)

            history['train_loss'].append(epoch_loss)
            history['train_f1'].append(epoch_f1)
            history['train_sent_acc'].append(epoch_sent_acc)

            print("Epoch ", epoch)
            print(
                f"\tTrain F1: {epoch_f1 * 100:.2f} | Train loss: {epoch_loss:.2f} | Sent acc: {epoch_sent_acc * 100: .2f}")

            self.model.eval()
            with torch.no_grad():
                epoch_loss_val = 0
                true_tags_epoch_val = []
                pred_tags_epoch_val = []
                val_iter = self.data.val_iter
                for batch in val_iter:
                    words = batch.word.to(self.device)
                    chars = batch.char.to(self.device)
                    true_tags = batch.tag.to(self.device)
                    pred_tags, batch_loss = self.model(words, chars, true_tags)
                    pred_tags_epoch_val += pred_tags
                    true_tags_epoch_val += [
                        [tag for tag in sent_tag if tag != self.data.tag_pad_idx]
                        for sent_tag in true_tags.permute(1, 0).tolist()
                    ]
                    epoch_loss_val += batch_loss.item()

            epoch_loss_val = epoch_loss_val / len(val_iter)
            epoch_f1_val = self.f1_score(pred_tags_epoch_val, true_tags_epoch_val)
            epoch_sent_acc_val = self.sent_accuracy(pred_tags_epoch_val, true_tags_epoch_val)

            lr_scheduling.step(epoch_f1_val)



            print(
                f"\tVal F1: {epoch_f1_val * 100:.2f} | Val loss: {epoch_loss_val:.2f} | Sent acc: {epoch_sent_acc_val * 100: .2f}")

            if epoch == N-1 and self.path:
            # if epoch > 0 and epoch_f1_val > max(history['val_f1']):
            # if epoch_f1_val > previous_f1:
            #     print(f"F1 score increases from {max(history['val_f1'])*100: .2f}% to {epoch_f1_val*100: .2f}%, saved model")
                print(f"model is saved in {self.path}")
                torch.save({
                    # 'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    # 'optimizer_state_dict': self.optimizer.state_dict(),
                    # 'loss': loss
                }, self.path)

            history['val_loss'].append(epoch_loss_val)
            history['val_f1'].append(epoch_f1_val)
            history['val_sent_acc'].append(epoch_sent_acc_val)
            # previous_f1 = epoch_f1_val

            print("-----------------------------------------------------")
        return history



