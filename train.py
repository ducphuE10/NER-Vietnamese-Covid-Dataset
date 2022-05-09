import torch

class Trainer(object):

    def __init__(self, model, data, optimizer_cls, loss_fn_cls):
        self.model = model
        self.data = data
        self.optimizer = optimizer_cls(model.parameters())
        self.loss_fn = loss_fn_cls(ignore_index=self.data.tag_pad_idx)

    def accuracy(self, preds, y):
        max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
        non_pad_elements = (y != self.data.tag_pad_idx).nonzero()  # prepare masking for paddings, != 0
        correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
        return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])

    def epoch(self):
        epoch_loss = 0
        epoch_acc = 0
        self.model.train()
        for batch in self.data.train_iter:
            text = batch.word  # text = [sent len, batch size]
            true_tags = batch.tag  # tags = [sent len, batch size]

            self.optimizer.zero_grad()
            pred_tags = self.model(text)
            # to calculate the loss and accuracy, we flatten both prediction and true tags
            # flatten pred_tags to [sent len, batch size, output dim]
            pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
            # flatten true_tags to [sent len * batch size]
            true_tags = true_tags.view(-1)
            batch_loss = self.loss_fn(pred_tags, true_tags)
            batch_loss.backward()
            self.model.set_embedding_weight_unk()

            batch_acc = self.accuracy(pred_tags, true_tags)
            self.optimizer.step()
            epoch_loss += batch_loss.item()
            epoch_acc += batch_acc.item()

        return epoch_loss / len(self.data.train_iter), epoch_acc / len(self.data.train_iter)

    def evaluate(self, iterator):
        epoch_loss = 0
        epoch_acc = 0
        self.model.eval()
        with torch.no_grad():
            # similar to epoch() but model is in evaluation mode and no backprop
            for batch in iterator:
                text = batch.word
                true_tags = batch.tag
                pred_tags = self.model(text)
                pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
                true_tags = true_tags.view(-1)
                batch_loss = self.loss_fn(pred_tags, true_tags)
                batch_acc = self.accuracy(pred_tags, true_tags)
                epoch_loss += batch_loss.item()
                epoch_acc += batch_acc.item()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    # main training sequence
    def train(self, n_epochs):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(n_epochs):
            self.epoch()
            print("Epoch ", epoch)
            train_loss, train_acc = self.epoch()
            print(f"\tTrn Loss: {train_loss:.3f} | Trn Acc: {train_acc * 100:.2f}%")
            val_loss, val_acc = self.evaluate(self.data.val_iter)
            print(f"\tVal Loss: {val_loss:.3f} | Val Acc: {val_acc * 100:.2f}%")
            print("----------------------------------------------------------------")

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

        return train_losses, val_losses, train_accuracies, val_accuracies