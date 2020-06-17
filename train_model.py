import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import glob
from Models import *
import sys
from data_utils import *
from skimage.io import imread, imsave
from sklearn.metrics import f1_score
import argparse




def lr_scheduler(optimizer, init_lr, epoch):


    for param_group in optimizer.param_groups:

        if  epoch == 70 or epoch == 105 or epoch==115 :
            param_group['lr']=param_group['lr']/10

        if epoch == 0:
            param_group['lr']=init_lr

        print('Current learning rate is {}'.format(param_group['lr']))


    return optimizer


def train(folder_names, val_folders, num_epochs , weight_path,ref_image_name,val_folder, gpu_no):

    best_val_acc=0.0
    model=network(ref_image_name, gpu_no)
    model= model.cuda()
    lrate=1e-3

    optimizer_s = optim.SGD(model.parameters(), lr=lrate, weight_decay=1e-2, momentum=0.9)

    criterion = torch.nn.CrossEntropyLoss()
    train_dataset = BasicDataset(folder_names)
    val_dataset = BasicDataset(val_folders, validation=True)
    dataset_loader = DataLoader(train_dataset,
                                             batch_size=64, shuffle=True, num_workers=8)
    test_loader = DataLoader(val_dataset,
                                        batch_size=64, shuffle=False, num_workers=8)
    dataset_test_len=len(val_dataset)
    dataset_train_len=len(train_dataset)

    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    if not os.path.exists('./stats'):
        os.makedirs('./stats')

    epochs= []
    train_acc=[]
    test_acc=[]
    train_loss=[]
    test_loss = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.
        all_labels = list()
        all_outputs = list()
        epochs.append(epoch)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        optimizer  = lr_scheduler(optimizer_s, lrate, epoch)
        print('*' * 70)
        running_loss = 0.0
        running_corrects = 0.0
        train_batch_ctr = 0.0
        GT_train=[]
        predicted_labels_train=[]
        for batch_num, (images, labels, _) in enumerate(dataset_loader):
            images = Variable(images.cuda(),requires_grad=True)
            labels = Variable(labels.cuda(),requires_grad=False)
            optimizer.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)

            loss = F.nll_loss(outputs, labels)

            loss.backward()
            optimizer.step()

            train_batch_ctr = train_batch_ctr + 1



            running_loss += loss.item()

            running_corrects += torch.sum(preds == labels.data)

            epoch_acc = float(running_corrects) / (dataset_train_len)

            GT_train.append(labels.cpu().data.numpy()[0])
            predicted_labels_train.append(preds.cpu().data.numpy()[0])


        print ('Train corrects: {} Train samples: {} Train accuracy: {}' .format( running_corrects, (dataset_train_len),epoch_acc))
        train_acc.append(epoch_acc)
        train_loss.append(running_loss / train_batch_ctr)





        model.eval()
        test_running_corrects = 0.0
        test_batch_ctr = 0.0
        test_running_loss = 0.0
        test_total = 0.0
        GT_test=[]
        predicted_labels_test=[]



        for (image, label,_)  in test_loader:



            with torch.no_grad():
                image, label = Variable(image.cuda()), Variable(label.cuda())
                test_outputs  = model(image)
                _, predicted_test = torch.max(test_outputs.data, 1)

                loss = F.nll_loss(test_outputs, label)


                test_running_loss += loss.item()
                test_batch_ctr = test_batch_ctr+1

                test_running_corrects += torch.sum(predicted_test == label.data)
                test_epoch_acc = float(test_running_corrects) / (dataset_test_len)
                GT_test.append(label.cpu().data.numpy()[0])
                predicted_labels_test.append(predicted_test.cpu().data.numpy()[0])

        if test_epoch_acc > best_val_acc:
            torch.save(model, args.weight_path + '_best.pt')
            best_val_acc=test_epoch_acc

        test_acc.append(test_epoch_acc)
        test_loss.append(test_running_loss / test_batch_ctr)

        print('Test corrects: {} Test samples: {} Test accuracy {}' .format(test_running_corrects,(dataset_test_len),test_epoch_acc))

        print('Train loss: {} Test loss: {}' .format(train_loss[epoch],test_loss[epoch]))


        print('*' * 70)


        plots(epochs, train_acc, test_acc, train_loss, test_loss, './plots/'+args.stats_file_name)
        write_csv('./stats/'+args.stats_file_name+'.csv', train_acc,test_acc,train_loss,test_loss, epoch)

    torch.save(model, args.weight_path + '_final.pt')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_folder', type=int, default=0, help='folder to be used as a validation set')
    parser.add_argument('--stats_file_name',type=str, default='stats',  help='file name to save training stats')
    parser.add_argument('--train_dir', type=str, default='train_data', help='train data directory')
    parser.add_argument('--weight_path', type=str, default='model', help='model path')
    parser.add_argument('--gpu', type=int, default='0', help='GPU number')
    parser.add_argument('--ref_name', type=str, default='./ref_all.bmp', help='reference image name')

    args, _ = parser.parse_known_args()
    torch.cuda.set_device(args.gpu)
  
    ref_image_name = args.ref_name
 
    folders=sorted(glob.glob(args.train_dir))
    train_folders=[folders[0], folders[1], folders[2],folders[3], folders[4], folders[5], folders[6]]
    del train_folders[args.val_folder]

    val_folders = [folders[args.val_folder]]


    print('training on:\n {}'.format(train_folders))
    print('validation on fold:\n {}'.format(val_folders))
    train(train_folders, val_folders, 120, args.weight_path,ref_image_name,args.val_folder, args.gpu)
