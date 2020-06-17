
from Models import *
from data_utils import *
import glob
import numpy as np
from sklearn.svm import SVC
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default='test_labels.csv', help='csv file name to save labels')
    parser.add_argument('--aux_mode', default=True, help='if True, aux classifier is also used for predictions ' )
    parser.add_argument('--CP', type=float, default=0.9, help='threshold for using aux classifier')
    parser.add_argument('--test_dir', type=str, default='test_data', help='test data directory')
    parser.add_argument('--train_dir', type=str, default='train_data', help='train data directory')
    parser.add_argument('--model_path', type=str, default='model.pt', help='saved model path')
    parser.add_argument('--gpu', type=int, default='0', help='GPU number')

    args, _ = parser.parse_known_args()

    model = torch.load(args.model_path, map_location='cuda:'+str(args.gpu))
    model= model.cuda(args.gpu)
    model.eval()

    if args.aux_mode:
        print('Using aux mode')
        def copy_data(m, i, o):
                embedding.copy_(o)

        folders = sorted(glob.glob(args.train_dir))

        val_folders=[folders[0], folders[1], folders[2],folders[3], folders[4], folders[5], folders[6]]

        val_dataset = BasicDataset(val_folders, validation=True)

        test_loader = DataLoader(val_dataset,
                                           batch_size=1, shuffle=False, num_workers=8)
        dataset_test_len=len(val_dataset)

        layer = model._modules.get('layers')
        layer=layer._modules.get('9')
        model.eval()
        test_running_corrects = 0.0
        test_batch_ctr = 0.0
        test_running_loss = 0.0
        test_total = 0.0




        index=0;
        extracted_features=[]
        initial_labels=[]
        true_labels=[]

        for (image, label,_)  in test_loader:

            embedding = torch.zeros(1,64, 10,10)
            def copy_data(m, i, o):
                embedding.copy_(o)
            h = layer.register_forward_hook(copy_data)

            image, label = Variable(image.cuda(args.gpu),volatile=True), Variable(label.cuda(args.gpu))
            test_outputs= model(image)

            h.remove()

            _, predicted_test = torch.max(test_outputs.data, 1)


            loss =  F.nll_loss(test_outputs, label)
            index+=1;


            embedding=embedding.squeeze(0)
            embedding=embedding.detach().numpy()
            embedding=np.reshape(embedding, (100,64))
            extracted_features.append(embedding)


            print('working on train image: {} '.format(index))

            initial_labels.append(predicted_test.cpu().data.numpy()[0])
            true_labels.append(label.cpu().data.numpy()[0])


        effective_labels=np.where(np.equal(initial_labels,true_labels))
        effective_features=np.array(extracted_features)[effective_labels[0]]
        effective_features_mean=(np.mean(effective_features,2))
        effective_true_labels=np.array(true_labels)[effective_labels[0]]

        print('training aux clf')
        aux_clf= SVC(C=10, gamma=100)
        aux_clf.fit(effective_features_mean,effective_true_labels)






    with open(args.file_name, 'w') as f:
                        f.write('{0}\n'.format('label'))



    folders = sorted(glob.glob(args.test_dir))
    val_folders = [folders[0]]
    print(val_folders)
    val_dataset = BasicDataset(val_folders, relative_path="/*", validation=True, testing=True)
    test_loader = DataLoader(val_dataset,
                                       batch_size=1, shuffle=False, num_workers=8)
    dataset_test_len=len(val_dataset)
    print(dataset_test_len)
    index=-1
    true_labels=[]

    initial_labels=np.random.rand(dataset_test_len,1)
    for (image, image_name)  in test_loader:
        index+=1

        set_gpu=1
        layer = model._modules.get('layers')
        layer=layer._modules.get('9')
        embedding = torch.zeros(1,64, 10,10)
        def copy_data(m, i, o):
            embedding.copy_(o)
        h = layer.register_forward_hook(copy_data)



        print('working on test image: {}'.format(index,dataset_test_len))

        with torch.no_grad():
                image = Variable(image.cuda(args.gpu))
                test_outputs= model(image)
                h.remove()

                _, predicted_test = torch.max(test_outputs.data, 1)
                initial_labels[index]=(predicted_test.cpu().data.numpy()[0])
                label_prob=torch.exp(test_outputs.data).cpu().data.numpy()[0][predicted_test.cpu().data.numpy()[0]]

                if args.aux_mode:
                    embedding=embedding.squeeze(0)
                    embedding=embedding.detach().numpy()
                    embedding=np.reshape(embedding, (100,64))
                    embedding_mean=np.mean(embedding,1)
                    embedding_mean=embedding_mean.reshape(1, -1)
                    if label_prob <=args.CP:
                        initial_labels[index]=aux_clf.predict(embedding_mean)


        with open(args.file_name, 'a') as f:
                         
                            f.write('{0}\n'.format(initial_labels[index][0]))
