#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sylvain
"""

import json
from pathlib import Path
import matplotlib
from tqdm import tqdm

matplotlib.use('Agg')


import datasets.VQADataset_Att as VQADataset
import torchvision.transforms as T
import torch
import numpy as np

import torch.utils.data
import os
import datetime

import wandb
from models import model_vit as model
from models import multitask_attention as multitask

def vqa_collate_fn(batch):
    # Separate the list of tuples into individual lists
    questions, answers, images, question_types, question_types_str = zip(*batch)

    # Convert tuples to appropriate tensor batches
    questions_batch = torch.stack(questions)
    answers_batch = torch.stack(answers)  
    images_batch = torch.stack(images)  
    question_types_batch = torch.tensor(question_types)
    # For question_types, you can choose whether to convert to tensor or leave as a list

    return questions_batch, answers_batch, images_batch, question_types_batch, question_types_str

def train(model, train_dataset, validate_dataset, batch_size, num_epochs, learning_rate, lr_fc, experiment_name, wandb_args, num_workers=4):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, persistent_workers=True, num_workers=num_workers, pin_memory=True, collate_fn=vqa_collate_fn)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=num_workers, pin_memory=True, collate_fn=vqa_collate_fn)
    
    # in the current implementation, we do not have shared parameters, except for the feature extractors (which are not trained)
    # if we want to share additional parameters, we need to add them to the shared_parameters() function in the model and uncomment the following line
    # as well as after the prediction step in the training loop
    #optimizer = torch.optim.Adam(model.shared_parameters(), lr=learning_rate)
    optimizer_heads = [torch.optim.Adam(classifier.parameters(), lr=lr_fc[i]) for i, classifier in enumerate(model.classifiers, 0)]
    criterion = torch.nn.CrossEntropyLoss()
    schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_head, patience=3, verbose=True, mode="max") for optimizer_head in optimizer_heads]

    # Create a directory for the experiment outputs
    output_dir = Path(f"outputs/{experiment_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store training parameters and metrics
    experiment_log = {
        "parameters": {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "modeltype": modeltype,
            "experiment_name": experiment_name
        },
        "final_results": {},
        "epoch_data": [],  
    }
    wandb.init(
        project="rsvitqa", 
        name=experiment_name,
        config=wandb_args
        )
    log_interval = wandb.config.get("log_interval")
    # magic
    wandb.watch(model, log_freq=log_interval)
        
    trainLoss = []
    valLoss = []

    accPerQuestionType = {'area': [], 'presence': [], 'count': [], 'comp': []}

    OA = []
    AA = []
    start_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        epoch_start_time = datetime.datetime.now()
        
        model = model.to("cuda")
        model.train()  # Switch to train mode
        runningLoss = 0.0

        # add tqdm to the training loader, providing a progress bar based on the number of batches
        progress_bar = tqdm(enumerate(train_loader, 0), total=len(train_loader), desc=f"Epoch {epoch}", position=0, leave=False)

        for i, data in progress_bar:
            question, answer, image, question_type, _ = data

            question = question.to("cuda")
            answer = answer.to("cuda")
            image = image.to("cuda")
            answer = answer.squeeze(1)
            question_type = question_type.to("cuda")

            pred = model(image, question, question_type)

            loss = criterion(pred, answer)
            
            samples_per_task = [(question_type == qt).sum().item() for qt in range(4)]

            #Calculate initial weights inversely proportional to class frequencies
            weights = [1.0 / (count if count > 0 else 1) for count in samples_per_task]

            #get factor to increase the weight of the harder classes from the config
            focus_increase_factors = wandb.config.get("focus_increase_factors")

            for qt in focus_increase_factors.keys(): 
                weights[int(qt)] *= focus_increase_factors[qt]

            #Normalize the weights
            weight_sum = sum(weights)
            normalized_weights = [weight / weight_sum for weight in weights] 
            if i % log_interval == 0:
                wandb.log({["weight_0", "weight_1", "weight_2", "weight_3"][qt]: normalized_weights[qt] for qt in range(4)})
                wandb.log({["samples_0", "samples_1", "samples_2", "samples_3"][qt]: samples_per_task[qt]/sum(samples_per_task) for qt in range(4)})

            loss_total = 0
            task_specific_losses = []
            
            # Forward pass for each task and collect individual losses
            for qt in range(4):
                mask = question_type == qt
                if mask.sum() == 0:
                    continue
                pred_qt = pred[mask]
                answer_qt = answer[mask]
                
                loss_qt = criterion(pred_qt, answer_qt)

                # Auxiliary loss for counting tasks
                if qt == 3:
                    # Convert predicted probabilities to count values
                    # using argmax to get the most likely count
                    predicted_counts = pred_qt.argmax(dim=1)
                    actual_counts = answer_qt

                    # Calculate auxiliary loss MAE
                    auxiliary_loss_qt = torch.nn.functional.l1_loss(predicted_counts.float(), actual_counts.float())

                    if i % log_interval == 0:
                        wandb.log({"epoch": epoch, "auxiliary_loss": auxiliary_loss_qt})

                    # Combine primary and auxiliary losses
                    #loss_qt * 0.8 += auxiliary_loss_qt * 0.2
                
                task_specific_losses.append(loss_qt * normalized_weights[qt])
                if i % log_interval == 0:
                    wandb.log({"epoch": epoch, f"loss_{qt}": loss_qt})
            
            # Now, sum the task-specific losses
            loss_total = sum(task_specific_losses)

            # Zero the gradients for the shared and task-specific parameters
            #optimizer.zero_grad()
            for optimizer_head in optimizer_heads:
                optimizer_head.zero_grad()

            # Backpropagate the total loss
            loss_total.backward()

            # Now, update all parameters
            for optimizer_head in optimizer_heads:
                optimizer_head.step()
            
            #optimizer.step()
            if i % log_interval == 0:
                wandb.log({"epoch": epoch, "loss": loss})
                wandb.log({"epoch": epoch, "loss_total": loss_total})

            # Update running loss and display it in the progress bar
            current_loss = loss.item() * question.size(0)
            runningLoss += current_loss
            progress_bar.set_postfix({'training_loss': '{:.6f}'.format(current_loss / len(data))})
        
            
        trainLoss.append(runningLoss / len(train_dataset))
        print('epoch #%d loss: %.3f' % (epoch, trainLoss[epoch]))
        model_save_path = output_dir / f"RSVQA_model_epoch_{epoch}.pth"
        torch.save(model.state_dict(), model_save_path)

        with torch.no_grad():
            model.eval()  # Make sure that the model is in evaluation mode
            runningLoss = 0.0
            
            # These dictionaries are used for detailed accuracy metrics
            countQuestionType = {'presence': 0, 'count': 0, 'comp': 0, 'area': 0}
            rightAnswerByQuestionType = {'presence': 0, 'count': 0, 'comp': 0, 'area': 0}

            # tqdm for the validation loop
            progress_bar = tqdm(enumerate(validate_loader, 0), total=len(validate_loader), desc="Validating", position=0, leave=False)

            for i, data in progress_bar:
                question, answer, image, question_type, question_type_str = data

                question = question.to("cuda")
                answer = answer.to("cuda")
                image = image.to("cuda")
                question_type = question_type.to("cuda")
                answer = answer.squeeze(1)  # Removing an extraneous dimension from the answers (added during saving)

                pred = model(image, question, question_type)
                loss = criterion(pred, answer)
                runningLoss += loss.item() * question.size(0)  # Accumulating the loss

                pred = np.argmax(pred.cpu().numpy(), axis=1)  # Getting the index of the max log-probability
                for j in range(answer.shape[0]):
                    countQuestionType[question_type_str[j]] += 1
                    if answer[j] == pred[j]:
                        rightAnswerByQuestionType[question_type_str[j]] += 1
            valLoss.append(runningLoss / len(validate_dataset))
            print('epoch #%d val loss: %.3f' % (epoch, valLoss[epoch]))
            wandb.log({"val_loss": valLoss[-1]})
            print(datetime.datetime.now())  

            question_type_to_idx = {
                "presence": 0,
                "comp": 1,
                "area": 2,
                "count": 3,
            }
        
            numQuestions = 0
            numRightQuestions = 0
            currentAA = 0
            for question_type in countQuestionType.keys():
                if countQuestionType[question_type] > 0:
                    accPerQuestiontype_tmp = rightAnswerByQuestionType[question_type] * 1.0 / countQuestionType[question_type]
                    accPerQuestionType[question_type].append(accPerQuestiontype_tmp)
                    wandb.log({"epoch": epoch, question_type: accPerQuestiontype_tmp})
                    print(f"{question_type}: {accPerQuestiontype_tmp}")
                    schedulers[question_type_to_idx[question_type]].step(accPerQuestiontype_tmp)
                    wandb.log({"epoch": epoch, f"lr_{question_type}": schedulers[question_type_to_idx[question_type]].optimizer.param_groups[0]['lr']})
                numQuestions += countQuestionType[question_type]
                numRightQuestions += rightAnswerByQuestionType[question_type]
                currentAA += accPerQuestionType[question_type][epoch]
                
        OA.append(numRightQuestions *1.0 / numQuestions)
        AA.append(currentAA * 1.0 / 4)
        wandb.log({"epoch": epoch, "OA": OA[-1], "AA": AA[-1]})
        print('OA: %.3f' % (OA[epoch]))
        print('AA: %.3f' % (AA[epoch]))
        epoch_end_time = datetime.datetime.now()
        epoch_info = {
        "epoch": epoch,
        "train_loss": trainLoss[-1],  
        "val_loss": valLoss[-1], 
        "OA": OA[-1],
        "AA": AA[-1],
        "start_time": epoch_start_time.strftime("%Y-%m-%d_%H:%M:%S"),
        "end_time": epoch_end_time.strftime("%Y-%m-%d_%H:%M:%S"),
        "total_time_in_hours": (epoch_end_time - epoch_start_time).total_seconds() / 3600,
        }
        experiment_log["epoch_data"].append(epoch_info)
    end_time = datetime.datetime.now()
    # Calculate and save final results or other relevant info
    experiment_log["final_results"] = {
        "average_train_loss": sum(trainLoss) / len(trainLoss),
        "average_val_loss": sum(valLoss) / len(valLoss),
        "start_time": start_time.strftime("%Y-%m-%d_%H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d_%H:%M:%S"),
        "total_time_in_hours": (end_time - start_time).total_seconds() / 3600
    }

    # Save the final experiment log
    final_log_file = output_dir / "final_experiment_log.json"
    with open(final_log_file, 'w') as outfile:
        json.dump(experiment_log, outfile, indent=4)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.autograd.set_detect_anomaly(True)
    disable_log = False
    learning_rate = 1e-5
    learning_rates = [1e-5, 1e-5, 2e-5, 1e-5]
    ratio_images_to_use = 1
    modeltype = 'ViT-Bert-Attention-Multitask-MUTAN'
    Dataset = 'HR'

    batch_size = 70
    num_epochs = 35
    patch_size = 512   
    num_workers = 12

    work_dir = os.getcwd()
    data_path = work_dir + '/data'
    images_path = data_path + '/image_representations_vit_att'
    questions_path = data_path + '/text_representations_bert' 
    questions_train_path = questions_path + '/train'
    questions_val_path = questions_path + '/val'
    experiment_name = f"{modeltype}_lr_{learning_rate}_batch_size_{batch_size}_run_{datetime.datetime.now().strftime('%m-%d_%H_%M')}"

    wandb_args = {
            "learning_rate": learning_rate,
            "learning_rates_fc": learning_rates,
            "ratio_images_to_use": ratio_images_to_use,
            "modeltype": modeltype,
            "Dataset": Dataset,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "patch_size": patch_size,
            "num_workers": num_workers,   
            "log_interval": 100,
            "experiment_name": experiment_name,
            "focus_increase_factors":  {}, # Increase the weight of the a question type by x
            "fusion_in": 1200,
            "fusion_hidden": 256,
            "fusion_hidden_2": 128,
        }

    train_dataset = VQADataset.VQADataset(questions_train_path, images_path)
    validate_dataset = VQADataset.VQADataset(questions_val_path, images_path) 
    
    # load from checkpoint
    mod = multitask.MultiTaskVQAModel().cuda()
    train(mod, train_dataset, validate_dataset, batch_size, num_epochs, learning_rate, learning_rates, experiment_name, wandb_args, num_workers)
    
    
