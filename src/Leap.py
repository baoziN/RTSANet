import torch
import argparse
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
from collections import defaultdict
import json
from models import Leap
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params

torch.manual_seed(1203)

model_name = 'LEAP'
resume_name = 'saved/LEAP'

if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_name, help='resume path')
parser.add_argument('--ddi', action='store_true', default=True, help="using ddi")
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--target_ddi', type=float, default=0.06, help="target ddi")
parser.add_argument('--T', type=float, default=2.0, help="T")
parser.add_argument('--decay_weight', type=float, default=0.85, help="decay weight")
parser.add_argument('--dim', type=int, default=64, help="dimension")
parser.add_argument('--cuda', type=int, default=1, help="which cuda")

args = parser.parse_args()

# def eval(model, data_eval, voc_size, epoch):
#     model.eval()

#     smm_record = []
#     ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
#     med_cnt, visit_cnt = 0, 0

#     for step, input in enumerate(data_eval):
#         y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

#         for adm_idx, adm in enumerate(input):
#             target_output = model(input[:adm_idx+1])

#             y_gt_tmp = np.zeros(voc_size[2])
#             y_gt_tmp[adm[2]] = 1
#             y_gt.append(y_gt_tmp)

#             # prediction prod
#             target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
#             y_pred_prob.append(target_output)

#             # prediction med set
#             y_pred_tmp = target_output.copy()
#             y_pred_tmp[y_pred_tmp >= 0.5] = 1
#             y_pred_tmp[y_pred_tmp < 0.5] = 0
#             y_pred.append(y_pred_tmp)

#             # prediction label
#             y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
#             y_pred_label.append(sorted(y_pred_label_tmp))
#             visit_cnt += 1
#             med_cnt += len(y_pred_label_tmp)

#         smm_record.append(y_pred_label)
#         adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))

#         ja.append(adm_ja)
#         prauc.append(adm_prauc)
#         avg_p.append(adm_avg_p)
#         avg_r.append(adm_avg_r)
#         avg_f1.append(adm_avg_f1)
#         llprint('\rtest step: {} / {}'.format(step, len(data_eval)))

#     # ddi rate
#     ddi_rate = ddi_rate_score(smm_record, path='data/output/ddi_A_final.pkl')

#     llprint('\nDDI Rate: {:.4}, Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
#         ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
#     ))

#     return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
def eval(model, data_eval, voc_size, epoch, max_visits=5):  
    model.eval()  

    smm_record = []  
    visit_true_labels = defaultdict(list)  
    visit_pred_labels = defaultdict(list)  
    visit_pred_probs = defaultdict(list)  

    med_cnt, visit_cnt = 0, 0  

    for step, input in enumerate(data_eval):  
        for adm_idx, adm in enumerate(input):  
            target_output = model(input[: adm_idx + 1])  

            y_gt_tmp = np.zeros(voc_size[2])  
            y_gt_tmp[adm[2]] = 1  

            target_output_np = F.sigmoid(target_output).detach().cpu().numpy()[0]  

            y_pred_tmp = target_output_np.copy()  
            y_pred_tmp[y_pred_tmp >= 0.5] = 1  
            y_pred_tmp[y_pred_tmp < 0.5] = 0  

            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]  

            if adm_idx < max_visits:  
                visit_true_labels[adm_idx].append(y_gt_tmp)  
                visit_pred_labels[adm_idx].append(y_pred_tmp)  
                visit_pred_probs[adm_idx].append(target_output_np)  

            visit_cnt += 1  
            med_cnt += len(y_pred_label_tmp)  

        smm_record.append([np.where(visit_pred_labels[adm_idx][-1]==1)[0].tolist() for adm_idx in range(min(len(input), max_visits))])  

    ddi_rate = ddi_rate_score(smm_record, path="data/output/ddi_A_final.pkl")  

    per_visit_metrics = dict()  
    for visit_idx in range(max_visits):  
        trues = np.array(visit_true_labels[visit_idx])  
        preds = np.array(visit_pred_labels[visit_idx])  
        probs = np.array(visit_pred_probs[visit_idx])  

        if len(trues) == 0:  
            continue  

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(trues, preds, probs)  
        per_visit_metrics[visit_idx + 1] = {  
            "Jaccard": float(adm_ja),  
            "PRAUC": float(adm_prauc),  
            "Precision": float(adm_avg_p),  
            "Recall": float(adm_avg_r),  
            "F1": float(adm_avg_f1),  
            "SampleNum": len(trues),  
        }  

    # 整体指标（所有访问合并）  
    all_trues = []  
    all_preds = []  
    all_probs = []  
    for visit_idx in range(max_visits):  
        all_trues.extend(visit_true_labels[visit_idx])  
        all_preds.extend(visit_pred_labels[visit_idx])  
        all_probs.extend(visit_pred_probs[visit_idx])  

    all_trues = np.array(all_trues)  
    all_preds = np.array(all_preds)  
    all_probs = np.array(all_probs)  

    overall_ja, overall_prauc, overall_p, overall_r, overall_f1 = multi_label_metric(all_trues, all_preds, all_probs)  

    print("\nDDI Rate: {:.4f}, Overall Jaccard: {:.4f}, PRAUC: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, AVG_MED: {:.4f}".format(  
        ddi_rate, overall_ja, overall_prauc, overall_p, overall_r, overall_f1, med_cnt / visit_cnt  
    ))  

    return ddi_rate, per_visit_metrics, med_cnt / visit_cnt 


def save_metrics(per_visit_metrics, ddi_rate, avg_med, epoch, save_dir="saved/metrics"):  
    os.makedirs(save_dir, exist_ok=True)  
    result_to_save = {  
        "ddi_rate": ddi_rate,  
        "avg_med": avg_med,  
        "per_visit_metrics": per_visit_metrics  
    }  
    save_path = os.path.join(save_dir, f"metrics_epoch_{epoch}.json")  
    # JSON保存，需要将所有值转换为可序列化类型  
    with open(save_path, "w") as f:  
        json.dump(result_to_save, f, indent=4)  
    print(f"Metrics saved to {save_path}") 


def save_best_metrics(per_visit_metrics, ddi_rate, avg_med, save_dir, best_ja):  
    os.makedirs(save_dir, exist_ok=True)  
    current_ja = per_visit_metrics.get(1, {}).get("Jaccard", 0)  

    if current_ja > best_ja[0]:  
        best_ja[0] = current_ja  # 更新最佳指标记录  
        result_to_save = {  
            "ddi_rate": ddi_rate,  
            "avg_med": avg_med,  
            "per_visit_metrics": per_visit_metrics  
        }  
        save_path = os.path.join(save_dir, "best_metrics.json")  
        with open(save_path, "w") as f:  
            json.dump(result_to_save, f, indent=4)  
        print(f"New best metrics saved to {save_path} with Jaccard={current_ja:.4f}")  
    else:  
        print(f"Current Jaccard={current_ja:.4f} did not improve. Best remains {best_ja[0]:.4f}")


def main():
    data_path = 'data/output/records_final.pkl'
    voc_path = 'data/output/voc_final.pkl'
    ddi_mask_path = 'data/output/ddi_mask_H.pkl'
    ehr_adj_path = 'data/output/ehr_adj_final.pkl'
    ddi_adj_path = 'data/output/ddi_A_final.pkl'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    ddi_mask_H = dill.load(open(ddi_mask_path, 'rb'))
    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    model = Leap(voc_size, ehr_adj, ddi_adj, ddi_mask_H, emb_dim=args.dim, device=device, ddi_in_memory=args.ddi)

    if args.Test:
        model.load_state_dict(torch.load(open(args.resume_path, 'rb')))
        model.to(device=device)
        tic = time.time()
        result = []
        # for _ in range(10):
        #     test_sample = np.random.choice(data_test, round(len(data_test) * 0.8), replace=True)
        #     ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, test_sample, voc_size, 0)
        #     result.append([ddi_rate, ja, avg_f1, prauc, avg_med])
        for _ in range(10):  
            test_sample = np.random.choice(data_test, round(len(data_test) * 0.8), replace=True)  
            ddi_rate, per_visit_metrics, avg_med = eval(model, test_sample, voc_size, 0)  

            # 这里示范取第1次访问的Jaccard和F1作为代表指标  
            ja = per_visit_metrics.get(1, {}).get("Jaccard", 0)  
            f1 = per_visit_metrics.get(1, {}).get("F1", 0)  
            prauc = per_visit_metrics.get(1, {}).get("PRAUC", 0)  

            result.append([ddi_rate, ja, f1, prauc, avg_med])
            
        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ''
        for m, s in zip(mean, std):
            outstring += '{:.4f} $\\pm$ {:.4f} & '.format(m, s)

        print(outstring)
        print('test time: {}'.format(time.time() - tic))
        return

    model.to(device=device)
    print('parameters', get_n_params(model))
    optimizer = Adam(list(model.parameters()), lr=args.lr)

    history = defaultdict(list)
    # best_epoch, best_ja = 0, 0
    best_epoch =  0
    best_ja = [0]

    EPOCH = 100
    for epoch in range(EPOCH):
        tic = time.time()
        print('\nepoch {} --------------------------'.format(epoch + 1))
        prediction_loss_cnt, neg_loss_cnt = 0, 0
        model.train()
        for step, input in enumerate(data_train):
            for idx, adm in enumerate(input):
                seq_input = input[:idx+1]
                loss_bce_target = np.zeros((1, voc_size[2]))
                loss_bce_target[:, adm[2]] = 1
                loss_bce_target = torch.FloatTensor(loss_bce_target).to(device)

                target_output1, loss_ddi = model(seq_input)

                loss_bce = F.binary_cross_entropy_with_logits(target_output1, loss_bce_target)

                if args.ddi:
                    target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
                    target_output1[target_output1 >= 0.5] = 1
                    target_output1[target_output1 < 0.5] = 0
                    y_label = np.where(target_output1 == 1)[0]
                    current_ddi_rate = ddi_rate_score([[y_label]], path='data/output/ddi_A_final.pkl')
                    if current_ddi_rate <= args.target_ddi:
                        loss = 0.9 * loss_bce + 0.1 * loss_ddi
                        prediction_loss_cnt += 1
                    else:
                        rnd = np.exp((args.target_ddi - current_ddi_rate) / args.T)
                        if np.random.rand(1) < rnd:
                            loss = loss_ddi
                            neg_loss_cnt += 1
                        else:
                            loss = 0.9 * loss_bce + 0.1 * loss_ddi
                            prediction_loss_cnt += 1
                else:
                    loss = loss_bce

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            llprint('\rtraining step: {} / {}'.format(step, len(data_train)))

        args.T *= args.decay_weight

        print()
        tic2 = time.time()
        # ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_eval, voc_size, epoch)
        ddi_rate, per_visit_metrics, avg_med = eval(model, data_eval, voc_size, epoch)  
        # save_metrics(per_visit_metrics, ddi_rate, avg_med, epoch)
        print('training time: {}, test time: {}'.format(time.time() - tic, time.time() - tic2))

        save_best_metrics(per_visit_metrics, ddi_rate, avg_med, save_dir=os.path.join("saved", args.model_name), best_ja=best_ja)
        
        ja = per_visit_metrics.get(1, {}).get("Jaccard", 0)  
        prauc = per_visit_metrics.get(1, {}).get("PRAUC", 0)  
        avg_p = per_visit_metrics.get(1, {}).get("Precision", 0)  
        avg_r = per_visit_metrics.get(1, {}).get("Recall", 0)  
        avg_f1 = per_visit_metrics.get(1, {}).get("F1", 0) 
        
        history['ja'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        history['med'].append(avg_med)

        if epoch >= 10:
            print('ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}'.format(
                np.mean(history['ddi_rate'][-5:]),
                np.mean(history['med'][-5:]),
                np.mean(history['ja'][-5:]),
                np.mean(history['avg_f1'][-5:]),
                np.mean(history['prauc'][-5:])
            ))

        # if epoch != 0 and best_ja < ja:
        if epoch != 0 and best_ja[0] < ja:
            best_epoch = epoch
            best_ja[0] = ja

        print('best_epoch: {}'.format(best_epoch))

    dill.dump(history, open(os.path.join('saved', args.model_name, 'history_{}.pkl'.format(args.model_name)), 'wb'))

if __name__ == '__main__':
    main()

