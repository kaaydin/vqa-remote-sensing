#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sylvain
"""

# Used to compute the accuracy of the model on the test set

import utils.VocabEncoder as VocabEncoder
import datasets.VQADataset_Att as VQADataset
from models import multitask_attention as model_multitask
from models import model as model_single
import matplotlib.pyplot as plt
import torch
from skimage import io
import numpy as np
import os
from tqdm import tqdm
import torch.utils.data.dataloader as dataloader
from sklearn.metrics import confusion_matrix
import seaborn as sns
        
def get_vocab():
    work_dir = os.getcwd()
    data_path = work_dir + '/data/text'
    allanswersJSON = os.path.join(data_path, 'USGSanswers.json')
    encoder_answers = VocabEncoder.VocabEncoder(allanswersJSON, questions=False, range_numbers = False)
        
    return encoder_answers.getVocab()

def load_model(experiment, type="multitask"):
    work_dir = os.getcwd()
    path = os.path.join(work_dir, experiment)
    if type == "multitask":
        network = model_multitask.MultiTaskVQAModel().cuda()
    else:
        network = model_single.VQAModel().cuda()
    state = network.state_dict()
    state.update(torch.load(path))
    network.load_state_dict(state)
    network.eval().cuda()
    return network

def get_image(image_id):
    work_dir = os.getcwd()
    images_path = os.path.join(work_dir + "/data/images", str(int(image_id)) + '.png')
    image = io.imread(images_path)
    return image

def load_dataset(text_path, images_path, batch_size=100, num_workers=6):
    test_dataset = VQADataset.VQADataset(text_path, images_path)
    test_loader = dataloader.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, persistent_workers=True, pin_memory=True, num_workers=num_workers)

    return test_loader

def run(network, text_path, images_path, experiment, dataset, num_batches=-1, save_output=False):
    test_dataset = VQADataset.VQADataset(text_path, images_path)
    test_loader = dataloader.DataLoader(test_dataset, batch_size=70, shuffle=False, persistent_workers=True, pin_memory=True, num_workers=6)
    
    print ('---' + experiment + '---')
    countQuestionType = {'area': 0, 'presence': 0, 'count': 0, 'comp': 0}
    rightAnswerByQuestionType = {'area': 0, 'presence': 0, 'count': 0, 'comp': 0}
    encoder_answers = get_vocab()
    confusionMatrix = np.zeros((len(encoder_answers), len(encoder_answers)))
    progress_bar = tqdm(enumerate(test_loader, 0), total=len(test_loader))
    count_preds = []
    count_answers = []
    count_absoulte_error = []
    count_mean_squared_error = []
    for i, data in progress_bar:
        if num_batches == 0:
            break
        num_batches -= 1
        question, answer, image, type_idx, type_str = data
        answer = answer.squeeze(1).to("cuda")
        question = question.to("cuda")
        image = image.to("cuda")

        pred = network(image,question, type_idx)
        
        answer = answer.cpu().numpy()
        pred = np.argmax(pred.cpu().detach().numpy(), axis=1)

        # decode answer and pred
        answers = [encoder_answers[a] for a in answer]
        preds = [encoder_answers[p] for p in pred]
        type_string = [t for t in type_str]

        for t, i in zip(type_string, range(len(type_string))):
            if t == "count":
                # check if answer is a number, if not, skip for a fair comparison since multitask model knows the question type
                try: 
                    temp = int(preds[i])
                except ValueError:
                    continue
                count_preds.append(int(preds[i]))
                count_answers.append(int(answers[i]))
                count_absoulte_error.append(abs(int(preds[i]) - int(answers[i])))
                count_mean_squared_error.append((int(preds[i]) - int(answers[i]))**2)

        for j in range(answer.shape[0]):
            countQuestionType[type_str[j]] += 1
            if answer[j] == pred[j]:
                rightAnswerByQuestionType[type_str[j]] += 1
            confusionMatrix[answer[j], pred[j]] += 1
    
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
    combined_list = [int(item) for item in count_preds + count_answers]
    unique_labels = np.unique(combined_list)
    sorted_labels = np.sort(unique_labels)
    cm = confusion_matrix(count_answers, count_preds, labels=sorted_labels)
    # convert confusion matrix to list and save
    confusionMatrix = cm.copy().tolist()
    # save confusion matrix
    np.save('confusion_matrix_' + experiment.split('/')[0], confusionMatrix)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Plotting the confusion matrix with annotations
    plt.figure(figsize=(24,20))
    sns.heatmap(cm_normalized, annot=False, fmt='d', xticklabels=sorted_labels, yticklabels=sorted_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_' + experiment.split('/')[0] + '.png', dpi=300, bbox_inches='tight')

    print(f"MAE: {np.mean(count_absoulte_error)}")
    print(f"MSE: {np.mean(count_mean_squared_error)}")
    print(f"Average count: {np.mean(count_preds)}")
    print(f"Average answer: {np.mean(count_answers)}")

    # second heatmap with not normalized confusion matrix
    plt.figure(figsize=(24,20))
    sns.heatmap(cm, annot=False, fmt='.2f', xticklabels=sorted_labels, yticklabels=sorted_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_' + experiment.split('/')[0] + '_abs.png', dpi=300, bbox_inches='tight')

    cm_log_scale = np.log(cm + 1)  # Adding 1 to avoid log(0)

    plt.figure(figsize=(24,20))
    sns.heatmap(cm_log_scale, annot=False, fmt="d", yticklabels=sorted_labels, xticklabels=sorted_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix (Log Scale)')
    plt.savefig('confusion_matrix_' + experiment.split('/')[0] + '_log.png', dpi=300, bbox_inches='tight')

    # Plotting individual distributions
    plt.figure(figsize=(12, 6))

    # Plot for 'answers'
    plt.subplot(1, 2, 1)
    plt.hist(count_answers, bins=range(0, 55), alpha=0.7, color='blue')
    plt.title('Ground Truth Distribution')
    plt.xlabel('Counts')
    plt.ylabel('Frequency')

    # Plot for 'preds'
    plt.subplot(1, 2, 2)
    plt.hist(count_preds, bins=range(2, 55), alpha=0.7, color='green')
    plt.title('Predictions Distribution')
    plt.xlabel('Counts')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('individual_distributions_' + experiment.split('/')[0] + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plotting overlapping distributions
    plt.figure(figsize=(6, 6))
    plt.hist(count_answers, bins=range(0, 55), alpha=0.5, label='Ground Truth', color='blue')
    plt.hist(count_preds, bins=range(0, 55), alpha=0.5, label='Predictions', color='green')
    plt.title('Overlapping Distributions')
    plt.xlabel('Counts')
    plt.ylabel('Frequency')
    plt.ylim(0, 40000)
    plt.legend()
    plt.savefig('overlapping_distributions_' + experiment.split('/')[1] + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    
    return Accuracies, confusionMatrix

if __name__ == '__main__':
    expes = {
            'HR': ['outputs/ViT-Bert-Attention-Multitask-MUTAN_lr_1e-05_batch_size_70_run_12-28_01_13/RSVQA_model_epoch_29.pth'],
    }
    work_dir = os.getcwd()
    data_path = work_dir + '/data'
    images_path = os.path.join(data_path, 'image_representations_vit_att')
    text_path = os.path.join(data_path, 'text_representations_bert/test_q_str')

    for dataset in expes.keys():
        acc = []
        mat = []
        for experiment_name in expes[dataset]:
            model_att = load_model(experiment_name, type="multitask")
            tmp_acc, tmp_mat = run(model_att, text_path, images_path, experiment_name, dataset)
            acc.append(tmp_acc)
            mat.append(tmp_mat)
            
        print('--- Total (' + dataset + ') ---')
        print('- Accuracies')
        for type_str in tmp_acc.keys():
            all_acc = []
            for tmp_acc in acc:
                all_acc.append(tmp_acc[type_str])
            print(' - ' + type_str + ': ' + str(np.mean(all_acc)) + ' ( stddev = ' + str(np.std(all_acc)) + ')')
