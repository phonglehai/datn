import argparse
from dataset import Dataset
from trainer import Trainer
from tester import Tester
from params import Params
import os

desc = 'Temporal KG Completion methods'
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('-dataset', help='Dataset', type=str, default='icews14', choices = ['icews14', 'icews05-15', 'gdelt'])
parser.add_argument('-model', help='Model', type=str, default='DE_DistMult', choices = ['DE_DistMult', 'DE_TransE', 'DE_SimplE','DE_SimplE_MLP'])
parser.add_argument('-ne', help='Number of epochs', type=int, default=600, choices = [600])
parser.add_argument('-bsize', help='Batch size', type=int, default=512, choices = [512])
parser.add_argument('-lr', help='Learning rate', type=float, default=0.001, choices = [0.001])
parser.add_argument('-reg_lambda', help='L2 regularization parameter', type=float, default=0.0, choices = [0.0])
parser.add_argument('-emb_dim', help='Embedding dimension', type=int, default=100, choices = [100])
parser.add_argument('-neg_ratio', help='Negative ratio', type=int, default=500, choices = [500])
parser.add_argument('-dropout', help='Dropout probability', type=float, default=0.4, choices = [0.0, 0.2, 0.4])
parser.add_argument('-save_each', help='Save model and validate each K epochs', type=int, default=50, choices = [50])
parser.add_argument('-se_prop', help='Static embedding proportion', type=float, default=0.36)

args = parser.parse_args()

dataset = Dataset(args.dataset)

params = Params(
    ne=args.ne,
    bsize=args.bsize,
    lr=args.lr,
    reg_lambda=args.reg_lambda,
    emb_dim=args.emb_dim,
    neg_ratio=args.neg_ratio,
    dropout=args.dropout,
    save_each=args.save_each,
    se_prop=args.se_prop
)

trainer = Trainer(dataset, params, args.model)
trainer.train()




validation_idx = list(range(
    args.save_each,
    args.ne + 1,
    args.save_each
))

best_mrr = -1.0
best_epoch = None

model_prefix = (
    "models/" + args.model + "/" + args.dataset + "/" +
    params.str_() + "_"
)

for epoch in validation_idx:
    model_path = model_prefix + str(epoch) + ".pth"

    if not os.path.exists(model_path):
        print("Skip missing:", model_path)
        continue

    tester = Tester(dataset, params, model_path, "valid")
    mrr = tester.test()

    if mrr > best_mrr:
        best_mrr = mrr
        best_epoch = epoch
print("Best epoch:", best_epoch)

best_model_path = model_prefix + str(best_epoch) + ".pth"
tester = Tester(dataset, params, best_model_path, "test")
tester.test()
