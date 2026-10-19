import csv
import nni
import time
import json
import math
import copy
import argparse
import warnings
import numpy as np

import torch
import torch.optim as optim
import torch_geometric
from torch.utils.data import DataLoader
from info_nce import InfoNCE, info_nce

from plot_G_degree import *
from utils5 import *
from models_mape import *
# from models5 import *
from dataloader5 import *
from torch_sparse import SparseTensor
from contrastlearning import contrast_loss
from degree import c_degree
from BS_ES_NS import *

warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def train(model, ppi_g, prot_embed, ppi_list, labels, index, batch_size, optimizer, loss_fn, epoch):
    f1_sum = 0.0
    loss_sum = 0.0
    fx_loss = 0

    batch_num = math.ceil(len(index) / batch_size)
    random.shuffle(index)

    model.train()
    Infoloss = InfoNCE().to(device)

    fc_loss = nn.MSELoss()
    # cl_loss = contrast_loss()
    loss1 = 0.0
    for batch in range(batch_num):
        if batch == batch_num - 1:
            train_idx = index[batch * batch_size:]
        else:
            train_idx = index[batch * batch_size: (batch + 1) * batch_size]

        # output, xa, xt = model(ppi_g, prot_embed, ppi_list, train_idx)
        output = model(ppi_g, prot_embed, ppi_list, train_idx)
        # print(xa[0].shape)  # torch.Size([1663, 1024])
        # print(output.shape)  # torch.Size([4427, 7])
        # loss1 = 0.0
        # for i in xa:
        #     loss1 += Infoloss(i.squeeze(),xt)
        # for i in xa:
        #     # loss = Infoloss(i,xt)
        # # loss = loss_fn(output, labels[train_idx]) + loss
        # for i in range(len(xa)):
        # loss1 += contrast_loss(prot_embed, xa[i], xt, batch_size) / 3
        # loss1 += fc_loss(xt, xa[i]) / 3
        # loss1 += Infoloss(xa[i // len(xa)], xa[(i + 1) // len(xa)]) / 3
        loss = loss_fn(output, labels[train_idx]) + loss1
        # loss = loss_fn(output, labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        f1_score = evaluat_metrics(output.detach().cpu(), labels[train_idx].detach().cpu())
        f1_sum += f1_score
        # metrics1 = Label_F1(valid_pre_result_list, valid_label_list, true_prob_list)
        # print("Epoch: {}, Batch: {}/{} | Train Loss: {:.5f}, F1-score: {:.5f}".format(epoch, batch+1, batch_num, loss.item(), f1_score))

    return loss_sum / batch_num, f1_sum / batch_num


def evaluator(model, ppi_g, prot_embed, ppi_list, labels, index, batch_size, mode='metric'):
    eval_output_list = []
    eval_labels_list = []

    batch_num = math.ceil(len(index) / batch_size)

    model.eval()
    BS_f1 = 0
    ES_f1 = 0
    NS_f1 = 0
    with torch.no_grad():
        for batch in range(batch_num):
            if batch == batch_num - 1:
                eval_idx = index[batch * batch_size:]
            else:
                eval_idx = index[batch * batch_size: (batch + 1) * batch_size]

            output = model(ppi_g, prot_embed, ppi_list, eval_idx)
            # output, xa, xt = model(ppi_g, prot_embed, ppi_list, eval_idx)
            eval_output_list.append(output.detach().cpu())
            eval_labels_list.append(labels[eval_idx].detach().cpu())

        # BS_list, ES_list, NS_list = BS_ES_NS(ppi_list, ppi_split_dict['train_index'], ppi_split_dict['val_index'],
        #                                      ppi_split_dict['test_index'])

        f1_score = evaluat_metrics(torch.cat(eval_output_list, dim=0), torch.cat(eval_labels_list, dim=0))
        # labels_f1 = label_metrics(torch.cat(eval_output_list, dim=0), torch.cat(eval_labels_list, dim=0))
        # if index == ppi_split_dict['test_index']:
        #     BS_f1, ES_f1, NS_f1 = BS_ES_NS_metrics(torch.cat(eval_output_list, dim=0),
        #                                            torch.cat(eval_labels_list, dim=0), BS_list, ES_list, NS_list)

    if mode == 'metric':
        # return f1_score, labels_f1, BS_f1, ES_f1, NS_f1
        return f1_score
    elif mode == 'output':
        return torch.cat(eval_output_list, dim=0), torch.cat(eval_labels_list, dim=0)


def pretrain_vae():
    if args.pre_train is None:
        protein_data, ppi_g, ppi_list, labels, ppi_split_dict = load_data(param['dataset'], param['split_mode'],
                                                                          param['seed'])
    else:
        protein_data = load_pretrain_data(args.pre_train)

    output_dir = "../results/{}/{}/{}/VAE/".format(param['dataset'], param['split_mode'], timestamp)
    check_writable(output_dir, overwrite=False)
    log_file = open(os.path.join(output_dir, "train_log.txt"), 'a+')
    with open(os.path.join(output_dir, "config.json"), 'a+') as tf:
        json.dump(param, tf, indent=2)

    vae_dataloader = DataLoader(protein_data, batch_size=128, shuffle=True, collate_fn=collate)
    vae_model = CodeBook(param, DataLoader(protein_data, batch_size=128, shuffle=False, collate_fn=collate)).to(device)
    vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=float(param['learning_rate']),
                                     weight_decay=float(param['weight_decay']))

    for epoch in range(1, param["pre_epoch"] + 1):
        for iter_num, batch_graph in enumerate(vae_dataloader):
            batch_graph = batch_graph.to(device)

            z, e, e_q_loss, recon_loss, mask_loss = vae_model(batch_graph)
            loss_vae = e_q_loss + recon_loss + mask_loss * param['mask_loss']

            vae_optimizer.zero_grad()
            loss_vae.backward()
            vae_optimizer.step()

            if (epoch - 1) % param['log_num'] == 0 and iter_num == 0:
                print(
                    "\033[0;30;43m Pre-training VQ-VAE | Epoch: {}, Batch: {} | Train Loss: {:.5f} | {:.5f} {:.5f} {:.5f}\033[0m".format(
                        epoch, iter_num, loss_vae.item(), e_q_loss.item(), recon_loss.item(), mask_loss.item()))
                log_file.write(
                    "Pre-training VQ-VAE | Epoch: {}, Batch: {} | Train Loss: {:.5f} | {:.5f} {:.5f} {:.5f}\n".format(
                        epoch, iter_num, loss_vae.item(), e_q_loss.item(), recon_loss.item(), mask_loss.item()))
                log_file.flush()

    torch.save(vae_model.state_dict(), os.path.join(output_dir, f'vae_model.ckpt'))

    del vae_model
    torch.cuda.empty_cache()


def main():
    protein_data, ppi_g, ppi_list, labels, ppi_split_dict = load_data(param['dataset'], param['split_mode'],
                                                                      param['seed'])
    print(param)
    # plot_G_sample(ppi_g)#绘制度的分布图
    # plot_G_histogram(ppi_g)
    # plot_G_bar(ppi_g)
    # exit()

    # BS_list,ES_list,NS_list = BS_ES_NS(ppi_list,ppi_split_dict['train_index'], ppi_split_dict['val_index'], ppi_split_dict['test_index'])

    # protein_data.to(device)
    vae_model = CodeBook(param, DataLoader(protein_data, batch_size=128, shuffle=False, collate_fn=collate)).to(device)
    if args.ckpt_path is None:
        vae_model.load_state_dict(
            torch.load(os.path.join("../results/{}/{}/{}/VAE/".format(param['dataset'], param['split_mode'], timestamp),
                                    f'vae_model.ckpt')))
    else:
        vae_model.load_state_dict(torch.load(args.ckpt_path))
    prot_embed = vae_model.Protein_Encoder.forward(vae_model.vq_layer).to(device)

    del vae_model
    torch.cuda.empty_cache()

    output_dir = "../results/{}/{}/{}/SEES_{}/".format(param['dataset'], param['split_mode'], timestamp, param['seed'])
    # print(output_dir)
    # exit()
    check_writable(output_dir, overwrite=False)
    log_file = open(os.path.join(output_dir, "train_log.txt"), 'a+')

    model = GIN(param).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(param['learning_rate']),
                                 weight_decay=float(param['weight_decay']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    es = 0
    val_best = 0
    test_val = 0
    test_best = 0
    best_epoch = 0
    labels_f1_best = [0] * 7
    # log_file.write('param:dataset={},split_mode={}\n'.format(param['dataset'],param['split_mode']))

    t1 = c_degree(ppi_g)
    # print(t1)
    # exit()
    for epoch in range(1, param["max_epoch"] + 1):

        # ppi_list_all = []
        # ppi_list_all.extend(ppi_split_dict['train_index'])
        # ppi_list_all.extend(ppi_split_dict['val_index'])
        # ppi_list_all.extend(ppi_split_dict['test_index'])

        # labelM = torch.load('../prodata/27klabel.pt').to(device)
        train_loss, train_f1_score = train(model, ppi_g, prot_embed, ppi_list, labels, ppi_split_dict['train_index'],
                                           param['batch_size'], optimizer, loss_fn, epoch)
        scheduler.step(train_loss)

        if (epoch - 1) % param['log_num'] == 0:

            # val_f1_score, val_labels_f1, BS_f1, ES_f1, NS_f1 = evaluator(model, ppi_g, prot_embed, ppi_list, labels,
            #                                                              ppi_split_dict['val_index'],
            #                                                              param['batch_size'], ppi_split_dict,
            #                                                              mode='metric')
            # test_f1_score, test_label_f1, BS_f1, ES_f1, NS_f1 = evaluator(model, ppi_g, prot_embed, ppi_list, labels,
            #                                                               ppi_split_dict['test_index'],
            #                                                               param['batch_size'], ppi_split_dict,
            #                                                               mode='metric')
            val_f1_score = evaluator(model, ppi_g, prot_embed, ppi_list, labels,
                                     ppi_split_dict['val_index'],
                                     param['batch_size'], mode='metric')
            test_f1_score = evaluator(model, ppi_g, prot_embed, ppi_list, labels, ppi_split_dict['test_index'],
                                      param['batch_size'], mode='metric')
            if test_f1_score > test_best:
                test_best = test_f1_score

            if val_f1_score >= val_best:
                val_best = val_f1_score
                test_val = test_f1_score
                # labels_f1_best = test_label_f1

                state = copy.deepcopy(model.state_dict())
                es = 0
                best_epoch = epoch
            else:
                es += 1
            # if epoch == 1:
            #     print("param".format(param))
            #     log_file.write(
            #         "param".format(param))
            print(
                "\033[0;30;46m Epoch: {}, Train Loss: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f} | Best Epoch: {}\033[0m".format(
                    epoch, train_loss, train_f1_score, val_f1_score, test_f1_score, val_best, test_val, test_best,
                    best_epoch))
            log_file.write(
                " Epoch: {}, Train Loss: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f} | Best Epoch: {}\n".format(
                    epoch, train_loss, train_f1_score, val_f1_score, test_f1_score, val_best, test_val, test_best,
                    best_epoch))
            # every label_f1
            # if epoch == 491:
            #     for i in range(len(labels_f1_best)):
            #         print("\033[0;30;46m Epoch: {}, label{}_F1: {:.5f}\033[0m".format(epoch, i, labels_f1_best[i]))
            #     print("\033[0;30;46m Epoch: {}, BS_F1: {:.5f},ES_F1: {:.5f},NS_F1: {:.5f}\033[0m".format(epoch, BS_f1,
            #                                                                                              ES_f1, NS_f1))

            log_file.flush()
            if es == 500:
                print("Early stopping!")
                break

    torch.save(state, os.path.join(output_dir, "model_state.pth"))
    log_file.close()

    model.load_state_dict(state)
    eval_output, eval_labels = evaluator(model, ppi_g, prot_embed, ppi_list, labels, ppi_split_dict['test_index'],
                                         param['batch_size'], 'output')

    np.save(os.path.join(output_dir, "eval_output.npy"), eval_output.detach().cpu().numpy())
    np.save(os.path.join(output_dir, "eval_labels.npy"), eval_labels.detach().cpu().numpy())

    # jsobj = json.dumps(ppi_split_dict)
    # with open(os.path.join(output_dir, "ppi_split_dict.json"), 'w') as f:
    #     f.write(jsobj)
    #     f.close()

    return test_f1_score, test_val, test_best, best_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--input_dim", type=int, default=7)
    parser.add_argument("--output_dim", type=int, default=7)
    parser.add_argument("--ppi_hidden_dim", type=int, default=1024)
    parser.add_argument("--prot_hidden_dim", type=int, default=256)
    parser.add_argument("--ppi_num_layers", type=int, default=2)
    parser.add_argument("--prot_num_layers", type=int, default=4)

    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--max_epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--dropout_ratio", type=float, default=0.0)

    parser.add_argument("--pre_epoch", type=int, default=50)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--num_embeddings", type=int, default=1024)
    parser.add_argument("--mask_ratio", type=float, default=0.2)
    parser.add_argument("--sce_scale", type=float, default=1.5)
    parser.add_argument("--mask_loss", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_num", type=int, default=10)
    parser.add_argument("--data_mode", type=int, default=1)
    parser.add_argument("--data_split_mode", type=int, default=0)

    parser.add_argument("--dataset", type=str, default="SHS27k")
    parser.add_argument("--split_mode", type=str, default="random")
    parser.add_argument("--pre_train", type=str, default="SHS27k")
    parser.add_argument("--ckpt_path", type=str, default='../results/27k-random/VAE/vae_model.ckpt')
    # parser.add_argument("--pre_train", type=str, default=None)
    # parser.add_argument("--ckpt_path", type=str, default=None)

    args = parser.parse_args()
    param = args.__dict__
    param.update(nni.get_next_parameter())
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S") + f"-%3d" % ((time.time() - int(time.time())) * 1000)
    if os.path.exists("../configs/param_configs.json"):
        param = json.loads(open("../configs/param_configs.json", 'r').read())[param['dataset']][param['split_mode']]

    if param['data_mode'] == 0:
        param['dataset'] = 'SHS27k'
    elif param['data_mode'] == 1:
        param['dataset'] = 'SHS148k'
    elif param['data_mode'] == 2:
        param['dataset'] = 'STRING'

    if param['data_split_mode'] == 0:
        param['split_mode'] = 'random'
    elif param['data_split_mode'] == 1:
        param['split_mode'] = 'bfs'
    elif param['data_split_mode'] == 2:
        param['split_mode'] = 'dfs'

    set_seed(param['seed'])
    if args.ckpt_path is None:
        pretrain_vae()
    test_acc, test_val, test_best, best_epoch = main()
    nni.report_final_result(test_val)

    outFile = open('../PerformMetrics_Metrics.csv', 'a+', newline='')
    writer = csv.writer(outFile, dialect='excel')
    results = [timestamp]
    for v, k in param.items():
        results.append(k)

    results.append(str(test_acc))
    results.append(str(test_val))
    results.append(str(test_best))
    writer.writerow(results)
