#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sylvain
"""

import json
from pathlib import Path
from tqdm import tqdm

import datasets.VQADataset_Att as VQADataset
import torch
import numpy as np

import torch.utils.data
import os
import datetime

import wandb
from models import model

def vqa_collate_fn(batch):
    # Separate the list of tuples into individual lists
    questions, answers, images, question_types_idx, question_types = zip(*batch)

    # Convert tuples to appropriate tensor batches
    questions_batch = torch.stack(questions)
    answers_batch = torch.stack(answers)  
    images_batch = torch.stack(images)  

    return questions_batch, answers_batch, images_batch, question_types

def train(model, train_dataset, validate_dataset, batch_size, num_epochs, learning_rate, experiment_name, wandb_args, num_workers=4):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True, pin_memory=True, collate_fn=vqa_collate_fn)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True, pin_memory=True, collate_fn=vqa_collate_fn)
    
    model = model.to("cuda")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

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
    model = model.to("cuda")
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
        
        model.train()  # Switch to train mode
        runningLoss = 0.0
        print(f'Starting epoch {epoch+1}/{num_epochs}')

        # add tqdm to the training loader, providing a progress bar based on the number of batches
        progress_bar = tqdm(enumerate(train_loader, 0), total=len(train_loader), desc=f"Epoch {epoch+1}", position=0, leave=False)

        for i, data in progress_bar:
            question, answer, image, _ = data

            question = question.to("cuda")
            answer = answer.to("cuda")
            image = image.to("cuda")

            answer = answer.squeeze(1)

            pred = model(image, question)
            loss = criterion(pred, answer)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % log_interval == 0:
                wandb.log({"epoch":  epoch, "loss": loss})
            # Update running loss and display it in the progress bar
            current_loss = loss.item()
            runningLoss += current_loss

            # update the progress bar with additional info
            progress_bar.set_postfix({'training_loss': '{:.6f}'.format(current_loss)})
        
            
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

            # tqdm for the validation loop, similar to the training loop
            progress_bar = tqdm(enumerate(validate_loader, 0), total=len(validate_loader), desc="Validating", position=0, leave=False)

            for i, data in progress_bar:
                question, answer, image, type_idx, type_str = data

                question = question.to("cuda")
                answer = answer.to("cuda")
                image = image.to("cuda")

                answer = answer.squeeze(1)  # Removing an extraneous dimension from the answers

                pred = model(image, question)
                loss = criterion(pred, answer)
                runningLoss += loss.item() * question.size(0)  # Accumulating the loss

                pred = np.argmax(pred.cpu().numpy(), axis=1)  # Getting the index of the max log-probability
                for j in range(answer.shape[0]):
                    countQuestionType[type_str[j]] += 1
                    if answer[j] == pred[j]:
                        rightAnswerByQuestionType[type_str[j]] += 1
            valLoss.append(runningLoss / len(validate_dataset))
            print('epoch #%d val loss: %.3f' % (epoch, valLoss[epoch]))
            wandb.log({"epoch": epoch, "val_loss": valLoss[-1]})
            print(datetime.datetime.now())  
        
            numQuestions = 0
            numRightQuestions = 0
            currentAA = 0
            for type_str in countQuestionType.keys():
                if countQuestionType[type_str] > 0:
                    accPerQuestiontype_tmp = rightAnswerByQuestionType[type_str] * 1.0 / countQuestionType[type_str]
                    accPerQuestionType[type_str].append(accPerQuestiontype_tmp)
                    wandb.log({"epoch": epoch, type_str: accPerQuestiontype_tmp})
                    print(f"{type_str}: {accPerQuestiontype_tmp}")
                numQuestions += countQuestionType[type_str]
                numRightQuestions += rightAnswerByQuestionType[type_str]
                currentAA += accPerQuestionType[type_str][epoch]
                
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
        # Save the JSON log file after each epoch
        epoch_log_file = output_dir / f"epoch_{epoch}_log.json"
        with open(epoch_log_file, 'w') as outfile:
            json.dump(epoch_info, outfile, indent=4)
    end_time = datetime.datetime.now()
    # Calculate and save final results or other relevant info
    experiment_log["final_results"] = {
        "average_train_loss": sum(trainLoss) / len(trainLoss),
        "average_val_loss": sum(valLoss) / len(valLoss),
        "OA-epochs": sum(OA) / len(OA),
        "AA-epochs": sum(AA) / len(AA),
        "OA-max": {
            "epoch": int(np.argmax(OA)),
            "value": np.max(OA)
        },
        "AA-max": {
            "epoch": int(np.argmax(AA)),
            "value": np.max(AA)
        },
        "start_time": start_time.strftime("%Y-%m-%d_%H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d_%H:%M:%S"),
        "total_time_in_hours": (end_time - start_time).total_seconds() / 3600
    }

    # Save the final experiment log
    final_log_file = output_dir / "final_experiment_log.json"
    with open(final_log_file, 'w') as outfile:
        json.dump(experiment_log, outfile, indent=4)
    wandb.finish()


if __name__ == '__main__':
    disable_log = False
    
    train_configs = [
            {
            'batch_size': 70,
            'num_epochs': 35,
            'learning_rate': 0.00001
            },
            # {
            # 'batch_size': 700,
            # 'num_epochs': 35,
            # 'learning_rate': 0.00001
            # },
            # {
            # 'batch_size': 700,
            # 'num_epochs': 35,
            # 'learning_rate': 0.0001
            # },
            # {
            # 'batch_size': 1400,
            # 'num_epochs': 35,
            # 'learning_rate': 0.0001
            # }
        ]

    
    modeltype = 'ViT-BERT-Attention-MUTAN'
    Dataset = 'HR'
    patch_size = 512   
    num_workers = 6


    for config in train_configs:
        batch_size = config['batch_size']
        num_epochs = config['num_epochs']
        learning_rate = config['learning_rate']

        work_dir = os.getcwd()
        data_path = work_dir + '/data'
        images_path = data_path + '/image_representations_vit_att'
        questions_path = data_path + '/text_representations_bert_att'
        questions_train_path = questions_path + '/train'
        questions_val_path = questions_path + '/val'
        experiment_name = f"{modeltype}_lr_{learning_rate}_batch_size_{batch_size}_run_{datetime.datetime.now().strftime('%m-%d_%H_%M')}"

        wandb_args = {
                "learning_rate": learning_rate,
                "modeltype": modeltype,
                "Dataset": Dataset,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "patch_size": patch_size,
                "num_workers": num_workers,
                "log_interval": 100,
                "experiment_name": experiment_name
            }
        
        train_dataset = VQADataset.VQADataset(questions_train_path, images_path)
        validate_dataset = VQADataset.VQADataset(questions_val_path, images_path) 
        
        RSVQA = model.VQAModel()
        train(RSVQA, train_dataset, validate_dataset, batch_size, num_epochs, learning_rate, experiment_name, wandb_args, num_workers)
    
    