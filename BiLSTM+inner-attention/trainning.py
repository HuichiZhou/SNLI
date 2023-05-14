import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#需要删除一些代码
from torch.utils.tensorboard import SummaryWriter
#"/home/rain/anaconda3/envs/snli_torch/lib/python3.7/site-packages/torch/utils/tensorboard/__init__.py", line 4,
#注释关于LooseVersion的

import torchtext.legacy as legacy
from bilstm_snli import encoder, inner_attention
# from lstm_snli import encoder, inner_attention

# import EarlyStopping 但是本身他并不存在 需要自己加入
from pytorchtools import EarlyStopping

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1,0"
import pickle
# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#使用legacy进行分词
inputs = legacy.data.Field(
    init_token="</s>", lower=True, batch_first=True, include_lengths=True
)
answers = legacy.data.Field(sequential=False)

# 使用legacy对数据进行分割
train_data, validation_data, test_data = legacy.datasets.SNLI.splits(
    text_field=inputs, label_field=answers
)
print('train_data:',train_data)

# 构建词表大小
inputs.build_vocab(train_data, min_freq=1, vectors="glove.840B.300d")
answers.build_vocab(train_data)
pretrained_embeddings = inputs.vocab.vectors
# with open('/home/rain/zfy/TextFooler-master/pretrained_embeddings.pickle', 'wb') as f:
#     pickle.dump(pretrained_embeddings, f)
#print('pretrained_embeddings:',pretrained_embeddings.size())#pretrained_embeddings: torch.Size([56221, 300])
def save_checkpoint(state, acc):
    filename=f"my_checkpoint_{acc}.pt"
    print("=>Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def check_accuracy(validation_iterator, model, criterion):
    val_losses = []
    val_accuracies = []
    k = 0
    with torch.no_grad():
        for val_batch_idx, val_batch in enumerate(validation_iterator):
            val_hyp, val_hyp_length = val_batch.hypothesis
            val_prem, val_prem_length = val_batch.premise
            val_target = val_batch.label - 1

            # # 将数字标签转换回文字标签
            # if(k < 3):
            #     label_vocab = answers.vocab
            #     text_labels = [label_vocab.itos[idx] for idx in val_target]
            #     print(val_target)
            #     # 输出文字标签
            #     print(text_labels)
            #     k += 1
            scores = model(val_prem, val_hyp, val_prem_length, val_hyp_length)
            loss = criterion(scores, val_target)
            # return the indices of each prediction
            _, predictions = scores.max(1)
            num_correct = float((predictions == val_target).sum())
            num_sample = float(predictions.size(0))
            val_losses.append(loss.item())
            val_accuracies.append(num_correct / num_sample)
    return val_losses, val_accuracies


num_classes = 3
embedding_size = 300
load_model = False

num_layer = 1
p = 0.25
learning_rates = [0.001]
batch_size = 128
hidden_sizes = [300]
num_epochs = 50

best_val_accuracy = 0.0  # 用于跟踪最佳验证准确率

for hidden_size in hidden_sizes:
    for learning_rate in learning_rates:
        step = 0
        setence_embedding = encoder(
            hidden_size,
            pretrained_embeddings,
            embedding_size,
            num_layer,
            p
        ).to(device)

        model = inner_attention(setence_embedding).to(device)
        (
            train_iterator,
            validation_iterator,
            test_iterator
        ) = legacy.data.BucketIterator.splits(
            (train_data, validation_data, test_data),
            batch_size = batch_size,
            device = device,
            sort_within_batch = True
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        writer = SummaryWriter(
            "runs/SNLI/\
            inner_attention_tensorboard/hidden_size_{} learning_rate_{}".format(
                hidden_size, learning_rate
            )
        )

        if load_model:
            load_checkpoint(torch.load("./bilstm_model/my_checkpoint.pth.tar"))

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=3, verbose=True)

        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        # to track the average training accuracy per epoch as the model trains
        avg_train_accuracy = []
        # to track the average validation accuracy per epoch as the model trains
        avg_valid_accuracy = []

        for epoch in range(num_epochs):
            model.train()
            train_losses = []
            train_accuracies = []

            # if (epoch+1)%5 == 0:
            #     checkpoint = {
            #         "state_dict":model.state_dict(),
            #         "optimizer":optimizer.state_dict()
            #     }
            #     save_checkpoint(checkpoint, )

            for batch_idx, batch in enumerate(train_iterator):
                optimizer.zero_grad()

                prem_sentences, prem_length = batch.premise
                hyp_sentences, hyp_length = batch.hypothesis
                
                target = batch.label - 1
                # print(prem_sentences, target)
                scores = model(
                    prem_sentences, hyp_sentences, prem_length, hyp_length
                )
                
                loss = criterion(scores, target)
                train_losses.append(loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

                _, predictions = scores.max(1)
                num_correct = (predictions == target).sum()
                running_train_acc = float(num_correct) / float(
                    hyp_sentences.shape[0]
                )
                train_accuracies.append(running_train_acc)

            writer.add_scalar(
                "Training Loss", np.mean(train_losses), global_step=step
            )

            writer.add_scalar(
                "Training Accuracy", np.mean(train_accuracies), global_step=step
            )
            # step += 1
            
            model.eval()
            # Check for the running accuracy of validation
            val_losses, val_accuracies = \
                check_accuracy(validation_iterator, model, criterion)

            writer.add_scalar(
                "Validation Loss", np.mean(val_losses), global_step=step
            )
            writer.add_scalar(
                "Validation Accuracy",
                np.mean(val_accuracies),
                global_step=step,
            )

            avg_train_losses.append(np.mean(train_losses))
            avg_valid_losses.append(np.mean(val_losses))

            avg_train_accuracy.append(np.mean(train_accuracies))
            avg_valid_accuracy.append(np.mean(val_accuracies))

            step += 1

            print_msg = (f'[{epoch+1}/{num_epochs+1}] ' +
                         f'train_accur: {np.mean(train_accuracies):.3f} ' +
                         f'valid_accur: {np.mean(val_accuracies):.3f}')

            print(print_msg)

            if np.mean(val_accuracies) > best_val_accuracy:
                best_val_accuracy = np.mean(val_accuracies)
                # Save the model checkpoint
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
                save_checkpoint(checkpoint, np.mean(val_accuracies))

            early_stopping(np.mean(val_losses), model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
