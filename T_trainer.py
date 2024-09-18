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
        if self.dataset_name == "chemhiv" or self.dataset_name == "ogbg-hiv" or self.dataset_name == "chempcba" or self.dataset_name == "ogbg-pcba" or self.dataset_name == "ogbg-ppa":
            self.num_graphs = len(self.data.datalist)
            labels = torch.tensor(self.data.labels, dtype=torch.long)
            self.features = self.data.features
            if self.dataset_name == "ogbg-pcba":
                self.all_subgraphs, self.max_neighbors = generate_all_subgraphs(self.data.datalist, level="ogbg-pcba")
            elif self.dataset_name == "ogbg-hiv":
                self.all_subgraphs, self.max_neighbors = generate_all_subgraphs(self.data.datalist, level="ogbg-hiv")
            else:
                self.all_subgraphs, self.max_neighbors = generate_all_subgraphs(self.data.datalist, level="graph")
        else:
            self.num_nodes = data.y.shape[0]
            data.y = data.y.squeeze()
            self.features = data.x
            labels=self.data.y
            self.all_subgraphs, self.max_neighbors = generate_all_subgraphs(self.data, level="node")

        self.shortest_distances = compute_shortest_distances(self.all_subgraphs, self.max_neighbors)
        if self.dataset_name == "ogbg-pcba":
            self.train_dataset, self.test_dataset, self.val_dataset = create_datasets(
                data=self.data,
                all_subgraphs=self.all_subgraphs,
                shortest_distances=self.shortest_distances,
                features=self.features,
                labels=labels,
                name="ogbg-pcba"
            )
        elif self.dataset_name == "ogbg-hiv":
            self.train_dataset, self.test_dataset, self.val_dataset = create_datasets(
                data=self.data,
                all_subgraphs=self.all_subgraphs,
                shortest_distances=self.shortest_distances,
                features=self.features,
                labels=labels,
                name="ogbg-hiv"
            )
        else:
            self.train_dataset, self.test_dataset, self.val_dataset = create_datasets(
                data=self.data,
                all_subgraphs=self.all_subgraphs,
                shortest_distances=self.shortest_distances,
                features=self.features,
                labels=labels
            )

        print("train mask: ", self.data.train_mask)
        print("valid mask: ", self.data.val_mask)
        print("test mask: ", self.data.test_mask)
        
        print("----------------------------")
        print("----------------------------")
        print("train dataset length: ", len(self.train_dataset))
        print("valid dataset length: ", len(self.val_dataset))
        print("test dataset length: ", len(self.test_dataset))
        print("----------------------------")
        print("----------------------------")
        

        self.train_loader = DataLoader(self.train_dataset, batch_size=cfg.gt.train.batch_size, shuffle=False, pin_memory=True, drop_last=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=cfg.gt.train.batch_size, shuffle=False, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=cfg.gt.train.batch_size, shuffle=False, pin_memory=True)

        # self.data = self.data.to(self.device)
        # self.features = self.features.to(self.device)
        if self.dataset_name == "chemhiv" or self.dataset_name == "ogbg-hiv" or self.dataset_name == "chempcba" or self.dataset_name == "ogbg-pcba" or self.dataset_name == "ogbg-ppa":
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
        self.ckpt = f"output/{self.dataset_name}/GraphT_bin_{self.gt_n_layers}_layers.pt" #change as needed
        self.stopper = EarlyStopping(
            patience=cfg.gnn.train.early_stop, path=self.ckpt) if cfg.gnn.train.early_stop > 0 else None
        if self.dataset_name == "chempcba" or self.dataset_name == "ogbg-pcba" or self.dataset_name == "chemhiv" or self.dataset_name == "ogbg-hiv":
            self.loss_func = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_func = torch.nn.CrossEntropyLoss()

        from GNNs.gnn_utils import Evaluator
        self._evaluator = Evaluator(name=self.dataset_name)
        if self.dataset_name == "chemhiv" or self.dataset_name == "ogbg-hiv":
            self.evaluator = lambda pred, labels: self._evaluator.eval(
                {"y_pred": torch.sigmoid(pred),
                "y_true": labels.view(-1, 1)}
            )["rocauc"]
        elif self.dataset_name == "chempcba" or self.dataset_name == "ogbg-pcba":
            self.evaluator = lambda pred, labels: self._evaluator.eval(
                {"y_pred": torch.sigmoid(pred),
                "y_true": labels}
            )["ap"]
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
        if self.dataset_name == "chemhiv":
            labels = batch['label'].float()
            loss = self.loss_func(logits.squeeze(), labels)
        elif  self.dataset_name == "ogbg-hiv":
            labels = batch['label'].float()
            loss = self.loss_func(logits, labels)
        else:
            labels = batch['label']
            loss = self.loss_func(logits, labels)
        loss.backward()
        self.optimizer.step()
        return self._get_train_output(logits, batch['label'], loss.item())

    def _get_train_output(self, logits, labels, loss):
        if self.dataset_name == "chemhiv" or self.dataset_name == "ogbg-hiv" or self.dataset_name == "chempcba" or self.dataset_name == "ogbg-pcba":
            return loss, (logits, labels)
        else:
            train_acc = self.evaluator(logits, labels)
            print("ppa train logits: ", logits)
            print("ppa train predictions: ", logits.argmax(dim=-1, keepdim=True))
            print("ppa train labels: ", labels)
            return loss, train_acc

    @torch.no_grad()
    def _evaluate(self, batch):
        self.model.eval()
        logits = self._forward(batch)
        batch['label'] = batch['label'].to(logits.device)
        return self._get_evaluate_output(logits, batch['label'])

    def _get_evaluate_output(self, logits, labels):
        if self.dataset_name == "chemhiv" or self.dataset_name == "ogbg-hiv" or self.dataset_name == "chempcba" or self.dataset_name == "ogbg-pcba":
            return logits, labels
        else:
            print("ppa eval logits: ", logits)
            print("ppa eval predictions: ", logits.argmax(dim=-1, keepdim=True))
            print("ppa eval labels: ", labels)
            acc = self.evaluator(logits, labels)
            return acc

    @time_logger
    def train(self):
        for epoch in range(self.epochs):
            t0, es_str = time(), ''
            train_loss, train_acc = self._train_epoch()
            val_acc = self._validate_epoch()

            if self.stopper is not None:
                es_flag, es_str = self.stopper.step(val_acc, self.model, epoch)
                if es_flag:
                    print(f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break

            if epoch % LOG_FREQ == 0:
                print(f'Epoch: {epoch}, Time: {time()-t0:.4f}, Loss: {train_loss:.4f}, TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, ES: {es_str}')

        if self.stopper is not None:
            self.model.load_state_dict(torch.load(self.stopper.path))

        return self.model

    def _train_epoch(self):
        all_logits, all_labels = [], []
        train_loss, train_acc = 0, 0
        for batch in self.train_loader:
            loss, output = self._train(batch)
            train_loss += loss
            if self.dataset_name == "chemhiv" or self.dataset_name == "ogbg-hiv" or self.dataset_name == "chempcba" or self.dataset_name == "ogbg-pcba":
                logits, labels = output
                all_logits.append(logits)
                all_labels.append(labels)
            else:
                train_acc += output

        train_loss /= len(self.train_loader)
        if self.dataset_name == "chemhiv" or self.dataset_name == "ogbg-hiv" or self.dataset_name == "chempcba" or self.dataset_name == "ogbg-pcba":
            train_acc = self.evaluator(torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0))
        else:
            train_acc /= len(self.train_loader)
        return train_loss, train_acc

    @torch.no_grad()
    def _validate_epoch(self):
        if self.dataset_name == "chemhiv" or self.dataset_name == "ogbg-hiv" or self.dataset_name == "chempcba" or self.dataset_name == "ogbg-pcba":
            all_logits, all_labels = [], []
            for batch in self.val_loader:
                logits, labels = self._evaluate(batch)
                all_logits.append(logits)
                all_labels.append(labels)
            return self.evaluator(torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0))
        else:
            val_acc = sum(self._evaluate(batch) for batch in self.val_loader)
            return val_acc / len(self.val_loader)

    @torch.no_grad()
    def eval_and_save(self):
        torch.save(self.model.state_dict(), self.ckpt)
        val_acc = self._validate_epoch()
        test_acc = self._test_epoch()
        print(f'GraphT+{self.dataset_name}+{self.gt_n_layers} ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n')
        return {'val_acc': val_acc, 'test_acc': test_acc}

    def _test_epoch(self):
        if self.dataset_name == "chemhiv" or self.dataset_name == "ogbg-hiv" or self.dataset_name == "chempcba" or self.dataset_name == "ogbg-pcba":
            all_logits, all_labels = [], []
            for batch in self.test_loader:
                logits, labels = self._evaluate(batch)
                all_logits.append(logits)
                all_labels.append(labels)
            return self.evaluator(torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0))
        else:
            test_acc = sum(self._evaluate(batch) for batch in self.test_loader)
            return test_acc / len(self.test_loader)



