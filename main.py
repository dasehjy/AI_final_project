import argparse

import torch
import numpy as np
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from data_process import get_train_data_loader, get_test_data_loader, calc_metrics
from model import AttentionAddModel, CatModel, AttentionCatModel

def init_argparse():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--model', default='AttentionCatModel', type=str)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epoch', default=10,type=int)
    args = parser.parse_args()
    return args

args = init_argparse()
print('args:', args)

device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("using GPU")
else:
    device = torch.device("cpu")
    print("using CPU")

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
torch.backends.cudnn.deterministic = True

def model_train():
    train_data_loader, valid_data_loader = get_train_data_loader()
    model = None
    if args.model == 'AttentionCatModel':
        model = AttentionCatModel.from_pretrained('./pretrained_model/bert-base-uncased')
    elif args.model == 'AttentionAddModel':
        model = AttentionAddModel.from_pretrained('./pretrained_model/bert-base-uncased')
    elif args.model == 'CatModel':
        model = CatModel.from_pretrained('./pretrained_model/bert-base-uncased')
    else:
        return
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(lr=args.lr, params=optimizer_grouped_parameters)
    criterion = CrossEntropyLoss()
    best_rate = 0
    print('START TRAINING: ')
    for epoch in range(args.epoch):
        total_loss = 0
        correct = 0
        total = 0
        target_list = []
        pred_list = []
        model.train()
        for idx, (guid, tag, image, text) in enumerate(train_data_loader):
            tag = tag.to(device)
            image = image.to(device)
            text = text.to(device)
            out = model(image_input=image, text_input=text)
            loss = criterion(out, tag)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item() * len(guid)
            pred = torch.max(out, 1)[1]
            total += len(guid)
            correct += (pred == tag).sum()

            target_list.extend(tag.cpu().tolist())
            pred_list.extend(pred.cpu().tolist())

        total_loss /= total
        print('EPOCH {:02d}: '.format(epoch + 1), end='')
        print('TRAIN LOSS:{:.6f} '.format(total_loss), end='')
        rate = correct / total * 100
        print('ACC_RATE:{:.2f}% '.format(rate), end='')
        metrics = calc_metrics(target_list, pred_list)
        print('F1 SCORE: {:.2f}% '.format(metrics[3] * 100))

        total_loss = 0
        correct = 0
        total = 0
        target_list = []
        pred_list = []
        model.eval()

        for guid, tag, image, text in valid_data_loader:
            tag = tag.to(device)
            image = image.to(device)
            text = text.to(device)
            out = model(image_input=image, text_input=text)
            loss = criterion(out, tag)
            total_loss += loss.item() * len(guid)
            pred = torch.max(out, 1)[1]
            total += len(guid)
            correct += (pred == tag).sum()
            target_list.extend(tag.cpu().tolist())
            pred_list.extend(pred.cpu().tolist())

        total_loss /= total
        print('EVAL LOSS:{:.6f} '.format(total_loss), end='')
        rate = correct / total * 100
        print('ACC_RATE:{:.2f}% '.format(rate), end='')
        metrics = calc_metrics(target_list, pred_list)
        print('F1 SCORE: {:.2f}%'.format(metrics[3] * 100))

        if rate > best_rate:
            best_rate = rate
            print('SAVING BEST ACC_RATE ON THE VALIDATION SET:{:.2f}%'.format(rate))
            torch.save(model.state_dict(),args.model + '.pth')
        print()
    print('END TRAINING')

def model_test():
    test_data_loader = get_test_data_loader()
    if args.model == 'AttentionCatModel':
        model = AttentionCatModel.from_pretrained('./pretrained_model/bert-base-uncased')
    elif args.model == 'AttentionAddModel':
        model = AttentionAddModel.from_pretrained('./pretrained_model/bert-base-uncased')
    elif args.model == 'CatModel':
        model = CatModel.from_pretrained('./pretrained_model/bert-base-uncased')
    else:
        return
    model.load_state_dict(torch.load(args.model + '.pth'))
    model.to(device)
    guid_list = []
    pred_list = []
    model.eval()

    for guid, tag, image, text in test_data_loader:
        image = image.to(device)
        text = text.to(device)
        out = model(image_input=image, text_input=text)

        pred = torch.max(out, 1)[1]
        guid_list.extend(guid)
        pred_list.extend(pred.cpu().tolist())

    pred_mapped = {
        0: 'negative',
        1: 'neutral',
        2: 'positive',
    }
    
    with open('result1.txt', 'w', encoding='utf-8') as f:
        f.write('guid,tag\n')
        for guid, pred in zip(guid_list, pred_list):
            f.write(f'{guid},{pred_mapped[pred]}\n')
        f.close()

if __name__ == "__main__":
    model_train()
    model_test()

        




