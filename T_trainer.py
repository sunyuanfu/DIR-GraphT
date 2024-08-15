import torch
import torch.nn.functional as F
from time import time
import numpy as np
torch.autograd.set_detect_anomaly(True)

from GNNs.gnn_utils import EarlyStopping
from data_utils.load import load_data, load_gpt_preds
from utils import time_logger

from Transformer.model import GraphTransformer
from data_utils.dgl_dataset import create_datasets
from graph import generate_all_subgraphs, compute_shortest_distances
from torch.utils.data import DataLoader

LOG_FREQ = 1


class GTTrainer():

    def __init__(self, cfg):
        self.seed = cfg.seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_name = cfg.dataset
        self.lm_model_name = cfg.lm.model.name
        self.epochs = cfg.gt.train.epochs
        self.gt_n_layers = cfg.gt.train.n_layers
        self.gt_dim_hidden = cfg.gt.train.dim_hidden
        self.gt_dim_qk = cfg.gt.train.dim_qk
        self.gt_dim_v = cfg.gt.train.dim_v
        self.gt_dim_ff = cfg.gt.train.dim_ff
        self.gt_n_heads = cfg.gt.train.n_heads
        self.gt_drop_input = cfg.gt.train.input_dropout_rate
        self.gt_dropout = cfg.gt.train.dropout_rate
        self.gt_dropmu = 0.0
        self.gt_lln_heads = cfg.gt.train.last_layer_n_heads
        self.lr = cfg.gt.train.lr
        self.weight_decay = cfg.gt.train.weight_decay


        print("Loading pretrained LM features for GraphTransformer ...")
        # LM_emb_path = f"prt_lm/{self.dataset_name}/{self.lm_model_name}-seed{self.seed}.emb"
        # print(f"LM_emb_path: {LM_emb_path}")
        # features = torch.from_numpy(np.array(
        #     np.memmap(LM_emb_path, mode='r',
        #                 dtype=np.float16,
        #                 shape=(self.num_nodes, 768)))
        # ).to(torch.float32)

        # self.features = features.to(self.device)

        data, num_classes = load_data(
                self.dataset_name, use_dgl=False, use_text=False, seed=self.seed)
        self.num_classes = num_classes
        self.data = data
        if self.dataset_name == "chemhiv" or self.dataset_name == "ogbg-ppa":
            self.num_graphs = len(self.data.datalist)
            labels = torch.tensor(self.data.labels, dtype=torch.long)
            self.features = self.data.features
            self.all_subgraphs, self.max_neighbors = generate_all_subgraphs(self.data.datalist, level="graph")
        else:
            self.num_nodes = data.y.shape[0]
            data.y = data.y.squeeze()
            self.features = data.x
            labels=self.data.y
            self.all_subgraphs, self.max_neighbors = generate_all_subgraphs(self.data, level="node")

        self.shortest_distances = compute_shortest_distances(self.all_subgraphs, self.max_neighbors)
        self.train_dataset, self.test_dataset, self.val_dataset = create_datasets(
            data=self.data,
            all_subgraphs=self.all_subgraphs,
            shortest_distances=self.shortest_distances,
            features=self.features,
            labels=labels
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=cfg.gt.train.batch_size, shuffle=True, pin_memory=True, drop_last=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=cfg.gt.train.batch_size, shuffle=False, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=cfg.gt.train.batch_size, shuffle=False, pin_memory=True)

        # self.data = self.data.to(self.device)
        # self.features = self.features.to(self.device)
        if self.dataset_name == "chemhiv":
            self.model = GraphTransformer(n_layers=self.gt_n_layers, dim_in=self.features.size(1), dim_out=self.num_classes, dim_hidden=self.gt_dim_hidden, 
                                        dim_qk=self.gt_dim_qk, dim_v=self.gt_dim_v, dim_ff=self.gt_dim_ff, n_heads=self.gt_n_heads, drop_input=self.gt_drop_input, 
                                        dropout=self.gt_dropout, drop_mu=self.gt_dropmu, last_layer_n_heads=self.gt_lln_heads,
                                        level="graph")
        else:
            self.model = GraphTransformer(n_layers=self.gt_n_layers, dim_in=self.features.size(1), dim_out=self.num_classes, dim_hidden=self.gt_dim_hidden, 
                                        dim_qk=self.gt_dim_qk, dim_v=self.gt_dim_v, dim_ff=self.gt_dim_ff, n_heads=self.gt_n_heads, drop_input=self.gt_drop_input, 
                                        dropout=self.gt_dropout, drop_mu=self.gt_dropmu, last_layer_n_heads=self.gt_lln_heads,
                                        level="node")
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        print(f"\nNumber of parameters: {trainable_params}")
        self.ckpt = f"output/{self.dataset_name}/GraphT.pt"
        self.stopper = EarlyStopping(
            patience=cfg.gnn.train.early_stop, path=self.ckpt) if cfg.gnn.train.early_stop > 0 else None
        self.loss_func = torch.nn.CrossEntropyLoss()

        from GNNs.gnn_utils import Evaluator
        self._evaluator = Evaluator(name=self.dataset_name)
        if self.dataset_name == "chemhiv":
            self.evaluator = lambda pred, labels: self._evaluator.eval(
                {"y_pred": F.softmax(pred, dim=-1)[:, 1].unsqueeze(-1),
                "y_true": labels.view(-1, 1)}
            )["acc"]
        else:
            self.evaluator = lambda pred, labels: self._evaluator.eval(
                {"y_pred": pred.argmax(dim=-1, keepdim=True),
                "y_true": labels.view(-1, 1)}
            )["acc"]

    def _forward(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        attn_score, logits = self.model(batch)  # small-graph
        return logits

    def _train(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        logits = self._forward(batch)
        batch['label'] = batch['label'].to(logits.device)
        loss = self.loss_func(logits, batch['label'])
        train_acc = self.evaluator(logits, batch['label'])
        loss.backward()
        self.optimizer.step()
        return loss.item(), train_acc

    @ torch.no_grad()
    def _evaluate(self, batch):
        self.model.eval()
        logits = self._forward(batch)
        batch['label'] = batch['label'].to(logits.device)
        acc = self.evaluator(logits, batch['label'])
        return acc

    @time_logger
    def train(self):
        for epoch in range(self.epochs):
            t0, es_str = time(), ''
            train_loss, train_acc = 0, 0
            for batch in self.train_loader:
                loss, acc = self._train(batch)
                train_loss += loss
                train_acc += acc
            train_loss /= len(self.train_loader)
            train_acc /= len(self.train_loader)

            val_acc = 0
            val_acc = sum(self._evaluate(batch) for batch in self.val_loader)
            val_acc /= len(self.val_loader)

            if self.stopper is not None:
                es_flag, es_str = self.stopper.step(val_acc, self.model, epoch)
                if es_flag:
                    print(f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break

            if epoch % LOG_FREQ == 0:
                print(
                    f'Epoch: {epoch}, Time: {time()-t0:.4f}, Loss: {train_loss:.4f}, TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, ES: {es_str}')

        if self.stopper is not None:
            self.model.load_state_dict(torch.load(self.stopper.path))

        return self.model

    @torch.no_grad()
    def eval_and_save(self):
        torch.save(self.model.state_dict(), self.ckpt)
        val_acc, test_acc = 0, 0

        test_acc = sum(self._evaluate(batch) for batch in self.test_loader)
        val_acc = sum(self._evaluate(batch) for batch in self.val_loader)

        val_acc /= len(self.val_loader)
        test_acc /= len(self.test_loader)

        print(f'GraphT+{self.dataset_name}+{self.gt_n_layers} ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n')
        res = {'val_acc': val_acc, 'test_acc': test_acc}
        return res
