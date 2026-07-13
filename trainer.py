
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import Dataset
from params import Params
from de_distmult import DE_DistMult
from de_transe import DE_TransE
from de_simple import DE_SimplE
from de_simple_mlp import DE_SimplE_MLP
from tester import Tester

class Trainer:
    def __init__(self, dataset, params, model_name):
        instance_gen = globals()[model_name]
        self.model_name = model_name
        # self.model = nn.DataParallel(instance_gen(dataset=dataset, params=params))
        model = instance_gen(dataset=dataset, params=params)
        model = model.cuda()                  # ⬅️ QUAN TRỌNG
        self.model = nn.DataParallel(model)

        self.dataset = dataset
        self.params = params

    def train(self, early_stop=False):
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.reg_lambda
        ) #weight_decay corresponds to L2 regularization

        loss_f = nn.CrossEntropyLoss()
        print(self.params.ne)
        for epoch in range(1, self.params.ne + 1):
            last_batch = False
            total_loss = 0.0
            start = time.time()

            while not last_batch:
                optimizer.zero_grad()

                heads, rels, tails, years, months, days = self.dataset.nextBatch(self.params.bsize, neg_ratio=self.params.neg_ratio)
                last_batch = self.dataset.wasLastBatch()

                scores = self.model(heads, rels, tails, years, months, days)

                ###Added for softmax####
                num_examples = int(heads.shape[0] / (1 + self.params.neg_ratio))
                scores_reshaped = scores.view(num_examples, self.params.neg_ratio+1)
                l = torch.zeros(num_examples).long().cuda()
                loss = loss_f(scores_reshaped, l)
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()

            print(time.time() - start)
            print("Loss in iteration " + str(epoch) + ": " + str(total_loss) + "(" + self.model_name + "," + self.dataset.name + ")")

            if epoch % self.params.save_each == 0:
                self.saveModel(epoch)




    def saveModel(self, chkpnt):
        print("Saving the model (state_dict)")
        directory = "models/" + self.model_name + "/" + self.dataset.name + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        path = directory + self.params.str_() + "_" + str(chkpnt) + ".pth"
        torch.save(self.model.module.state_dict(), path)
