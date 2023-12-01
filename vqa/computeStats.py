#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sylvain
"""

# Calcul des statistiques sur un jeu de test

import VocabEncoder
import VQADataset
from models import model_vit as model
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torchvision.transforms as T
from torch.autograd import Variable
from skimage import io
import numpy as np
import pickle
import os
from tqdm import tqdm

def do_confusion_matrix(all_mat, old_vocab, new_vocab, dataset):
    print(new_vocab)
    new_mat = np.zeros((len(new_vocab), len(new_vocab)))
    for i in range(1,all_mat.shape[0]):
        answer = old_vocab[i]
        new_i = new_vocab.index(answer)
        for j in range(1,all_mat.shape[1]):
            answer = old_vocab[j]
            new_j = new_vocab.index(answer)
            new_mat[new_i, new_j] = all_mat[i, j]

    if len(old_vocab) > 20:#HR
        new_mat = new_mat[0:18,0:18]
        new_vocab = new_vocab[0:18]
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    cax = ax.matshow(np.log(new_mat+1), cmap="YlGn")
    #plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + new_vocab)
    ax.set_yticklabels([''] + new_vocab)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    fig.savefig('confusion_matrix_' + dataset + '.svg')
    #plt.close()

        

def get_vocab(dataset):
    work_dir = os.getcwd()
    data_path = work_dir + '/data/text'
    if dataset == "LR":
        allanswersJSON = os.path.join(data_path, 'answers.json')
        encoder_answers = VocabEncoder.VocabEncoder(allanswersJSON, questions=False, range_numbers = True)
    else:
        allanswersJSON = os.path.join(data_path, 'USGSanswers.json')
        encoder_answers = VocabEncoder.VocabEncoder(allanswersJSON, questions=False, range_numbers = False)
        
    return encoder_answers.getVocab()

def load_model(experiment, epoch, patch_size=512):
    weight_file = experiment + '_' + str(epoch) + '.pth'
    work_dir = os.getcwd()
    path = os.path.join(work_dir, 'outputs', weight_file)
    network = model.VQAModel().cuda()#input_size = patch_size).cuda()
    state = network.state_dict()
    state.update(torch.load(path))
    network.load_state_dict(state)
    network.eval().cuda()
    return network

def load_dataset(text_path, images_path, batch_size=100, num_workers=0):
    test_dataset = VQADataset.VQADataset(text_path, images_path)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_loader

def run(network, test_loader, experiment, dataset, num_batches=-1, save_output=False):
    
    batch_size = 100
    patch_size = 512
    
    print ('---' + experiment + '---')
    if dataset == 'LR':
        countQuestionType = {'rural_urban': 0, 'presence': 0, 'count': 0, 'comp': 0}
        rightAnswerByQuestionType = {'rural_urban': 0, 'presence': 0, 'count': 0, 'comp': 0}
    else:
        countQuestionType = {'area': 0, 'presence': 0, 'count': 0, 'comp': 0}
        rightAnswerByQuestionType = {'area': 0, 'presence': 0, 'count': 0, 'comp': 0}
    encoder_answers = get_vocab(dataset)
    confusionMatrix = np.zeros((len(encoder_answers), len(encoder_answers)))
    progress_bar = tqdm(enumerate(test_loader, 0), total=len(test_loader))
    for i, data in progress_bar:
        if num_batches == 0:
            break
        num_batches -= 1
        question, answer, image, type_str = data
        answer = answer.squeeze(1).to("cuda")
        question = question.to("cuda")
        image = image.to("cuda")

        pred = network(image,question)
        
        answer = answer.cpu().numpy()
        pred = np.argmax(pred.cpu().detach().numpy(), axis=1)
        
        for j in range(answer.shape[0]):
            countQuestionType[type_str[j]] += 1
            if answer[j] == pred[j]:
                rightAnswerByQuestionType[type_str[j]] += 1
            confusionMatrix[answer[j], pred[j]] += 1
            
        # if save_output:
        #     out_path = os.path.join(work_dir, 'output')
        #     if not os.path.exists(out_path):
        #         os.mkdir(out_path)
        #     for j in range(batch_size):
        #         viz_img = T.ToPILImage()(image_original[j].float().data.cpu())
        #         viz_question = encoder_questions.decode(question[j].data.cpu().numpy())
        #         viz_answer = encoder_answers.decode([answer[j]])
        #         viz_pred = encoder_answers.decode([pred[j]])
            
        #         imname = str(i * batch_size + j) + '_q_' + viz_question + '_gt_' + viz_answer + '_pred_' + viz_pred + '.png'
        #         # replace special characters
        #         imname = imname.replace('?', '')
        #         plt.imsave(os.path.join(out_path, imname), viz_img)
    
    Accuracies = {'AA': 0}
    for type_str in countQuestionType.keys():
        Accuracies[type_str] = rightAnswerByQuestionType[type_str] * 1.0 / countQuestionType[type_str]
        Accuracies['AA'] += Accuracies[type_str] / len(countQuestionType.keys())
    Accuracies['OA'] = np.trace(confusionMatrix)/np.sum(confusionMatrix)
    
    print('- Accuracies')
    for type_str in countQuestionType.keys():
        print (' - ' + type_str + ': ' + str(Accuracies[type_str]))
    print('- AA: ' + str(Accuracies['AA']))
    print('- OA: ' + str(Accuracies['OA']))
    
    return Accuracies, confusionMatrix

expes = {
         'HR': ['RNN_ViT_MUTAN_lr_1e-05_batch_size_700_run_11-23_12_40/RSVQA_model_epoch'],
         #'HRPhili': ['RSVQA_ViT-CLS_RNN_512_100_35_0.00001_HR_2023-30-10/RSVQA_model_epoch'],
}
work_dir = os.getcwd()
data_path = work_dir + '/data'

images_path = os.path.join(data_path, 'image_representations_vit')
text_path = os.path.join(data_path, 'text_representations/test')
test_loader = load_dataset(text_path, images_path, batch_size=100, num_workers=0)
for dataset in expes.keys():
    acc = []
    mat = []
    for experiment_name in expes[dataset]:
        if not os.path.isfile('accuracies_' + dataset + '_' + experiment_name + '.npy'):
            if dataset[-1] == 's':
                tmp_acc, tmp_mat = run(experiment_name, dataset[:-1], shuffle=True)
            else:
                model = load_model(experiment_name, 34)
                tmp_acc, tmp_mat = run(model, test_loader, experiment_name, dataset)
            # np.save('accuracies_' + dataset + '_' + experiment_name, tmp_acc)
            # np.save('confusion_matrix_' + dataset + '_' + experiment_name, tmp_mat)
        else:
            tmp_acc = np.load('accuracies_' + dataset + '_' + experiment_name + '.npy', allow_pickle=True)[()]
            tmp_mat = np.load('confusion_matrix_' + dataset + '_' + experiment_name + '.npy', allow_pickle=True)[()]
        acc.append(tmp_acc)
        mat.append(tmp_mat)
        
    print('--- Total (' + dataset + ') ---')
    print('- Accuracies')
    for type_str in tmp_acc.keys():
        all_acc = []
        for tmp_acc in acc:
            all_acc.append(tmp_acc[type_str])
        print(' - ' + type_str + ': ' + str(np.mean(all_acc)) + ' ( stddev = ' + str(np.std(all_acc)) + ')')
    
    if dataset[-1] == 's':
        vocab = get_vocab(dataset[:-1])
    else:
        vocab = get_vocab(dataset)

    all_mat = np.zeros(tmp_mat.shape)    
    for tmp_mat in mat:
        all_mat += tmp_mat
    
    if dataset[0] == 'H':
        new_vocab = ['yes', 'no', '0m2', 'between 0m2 and 10m2', 'between 10m2 and 100m2', 'between 100m2 and 1000m2', 'more than 1000m2'] + [str(i) for i in range(90)]
    else:
        new_vocab = ['yes', 'no', 'rural', 'urban', '0', 'between 0 and 10', 'between 10 and 100', 'between 100 and 1000', 'more than 1000']
        
    do_confusion_matrix(all_mat, vocab, new_vocab, dataset)


#labels = ['Yes', 'No', '<=10', '0', '<=100', '<=1000', '>1000', 'Rural', 'Urban']
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(np.log(confusionMatrix[1:,1:] + 1), cmap="YlGn")
##plt.title('Confusion matrix of the classifier')
#fig.colorbar(cax)
#ax.set_xticklabels([''] + labels)
#ax.set_yticklabels([''] + labels)
#plt.xlabel('Predicted')
#plt.ylabel('True')
#plt.show()
#fig.savefig(os.path.join(baseFolder, 'AccMatrix.pdf'))
#print(Accuracies)
