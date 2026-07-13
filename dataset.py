import random
import numpy as np
from random import shuffle
from collections import defaultdict
from scripts import shredFacts


class Dataset:
    """Implements the specified dataloader"""
    def __init__(self, ds_name):
        self.name = ds_name
        self.ds_path = "datasets/" + ds_name.lower() + "/"

        self.ent2id = {}
        self.rel2id = {}

        self.data = {
            "train": self.readFile(self.ds_path + "train.txt"),
            "valid": self.readFile(self.ds_path + "valid.txt"),
            "test":  self.readFile(self.ds_path + "test.txt")
        }

        self.start_batch = 0

        self.convertTimes()

        self.all_facts_as_tuples = set(
            tuple(d) for d in self.data["train"]
            + self.data["valid"]
            + self.data["test"]
        )

        for spl in ["train", "valid", "test"]:
            self.data[spl] = np.array(self.data[spl])


        self.rel2ents = defaultdict(set)
        for h, r, t, _, _, _ in self.data["train"]:
            self.rel2ents[r].add(h)
            self.rel2ents[r].add(t)

    # --------------------------------------------------

    def readFile(self, filename):
        with open(filename, "r") as f:
            data = f.readlines()

        facts = []
        for line in data:
            h, r, t, ts = line.strip().split("\t")
            facts.append([
                self.getEntID(h),
                self.getRelID(r),
                self.getEntID(t),
                ts
            ])
        return facts

    def convertTimes(self):
        for split in ["train", "valid", "test"]:
            for i, fact in enumerate(self.data[split]):
                date = list(map(float, fact[-1].split("-")))
                self.data[split][i] = fact[:-1] + date

    def numEnt(self):
        return len(self.ent2id)

    def numRel(self):
        return len(self.rel2id)

    def getEntID(self, ent_name):
        if ent_name not in self.ent2id:
            self.ent2id[ent_name] = len(self.ent2id)
        return self.ent2id[ent_name]

    def getRelID(self, rel_name):
        if rel_name not in self.rel2id:
            self.rel2id[rel_name] = len(self.rel2id)
        return self.rel2id[rel_name]


    def nextPosBatch(self, batch_size):
        if self.start_batch + batch_size > len(self.data["train"]):
            ret = self.data["train"][self.start_batch:]
            self.start_batch = 0
        else:
            ret = self.data["train"][self.start_batch:self.start_batch + batch_size]
            self.start_batch += batch_size
        return ret


    def addNegFacts2(self, bp_facts, neg_ratio):
        pos_neg_group_size = 1 + neg_ratio
        facts1 = np.repeat(np.copy(bp_facts), pos_neg_group_size, axis=0)
        facts2 = np.copy(facts1)

        rand_nums1 = np.random.randint(1, self.numEnt(), size=facts1.shape[0])
        rand_nums2 = np.random.randint(1, self.numEnt(), size=facts2.shape[0])

        for i in range(facts1.shape[0] // pos_neg_group_size):
            rand_nums1[i * pos_neg_group_size] = 0
            rand_nums2[i * pos_neg_group_size] = 0

        facts1[:, 0] = (facts1[:, 0] + rand_nums1) % self.numEnt()
        facts2[:, 2] = (facts2[:, 2] + rand_nums2) % self.numEnt()

        return np.concatenate((facts1, facts2), axis=0)


    def addHardNegFacts(self, bp_facts, neg_ratio, hard_ratio):
        pos_neg_group_size = 1 + neg_ratio
        facts = np.repeat(np.copy(bp_facts), pos_neg_group_size, axis=0)

        num_hard = int(neg_ratio * hard_ratio)

        for i in range(bp_facts.shape[0]):
            base = i * pos_neg_group_size
            r = bp_facts[i][1]

            for j in range(1, num_hard + 1):
                idx = base + j
                facts[idx, 2] = random.choice(
                    list(self.rel2ents[r])
                )

        return facts


    def nextBatch(self, batch_size, neg_ratio=1, hard_ratio=0.0):
        bp_facts = self.nextPosBatch(batch_size)

        if hard_ratio > 0:
            batch = self.addHardNegFacts(bp_facts, neg_ratio, hard_ratio)
        else:
            batch = self.addNegFacts2(bp_facts, neg_ratio)

        return shredFacts(batch)

    def wasLastBatch(self):
        return self.start_batch == 0
