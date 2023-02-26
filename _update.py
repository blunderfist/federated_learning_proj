import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings('always')


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, weights, model):
        self.args = args
        self.logger = logger
        self.global_weights = weights
        # self.dataloader = dataloader
        # self.trainloader, self.validloader, self.testloader = self.train_val_test(
        #     dataset, list(idxs))
        self.device = 'mps' #if args.gpu else 'cpu'
        # self.device = 'cpu'# if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion =torch.nn.CrossEntropyLoss().to(self.device)
        self.model = model

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self,  dataloader, global_round):
        # Set mode to train model
        self.model.train()
        epoch_loss = []
        epoch_acc = []
        train_correct = 0
        

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr = self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adamx':
            optimizer = torch.optim.Adamax(self.model.parameters(), lr = self.args.lr)

        for iter in range(self.args.local_ep):
            batch_loss = []
            acc = 0
            total=0
            for batch_idx, (images, labels) in enumerate(dataloader):
               
                images, labels = images.to(self.device), labels.to(self.device)
                
                total += len(images)

                self.model.zero_grad()
                output = self.model(images)
                loss = self.criterion(output,labels)
                loss.backward()
                optimizer.step()
                scores, predictions = torch.max(output.data, 1)
                # train_loss += loss.item() * images.size(0)

                
                # self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
                train_correct += (predictions == labels).sum().item()
                acc += (predictions == labels).sum().item()
                
            acc = acc / total
                
            # if self.args.verbose and (batch_idx % 10 == 0):
            print('| Global Round : {} | Local Epoch : {} |\tLoss: {:.6f} | Acc: {:.6f}'.format(
                    global_round, iter, loss.item(), acc*100))
                
                
            # train_acc = 100*train_correct / len(images)*len(dataloader)
            epoch_acc.append(acc)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
                
            
               
        
            
            
            

        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)

    def inference(self, model, testloader):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


