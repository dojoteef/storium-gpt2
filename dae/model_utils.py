'''

    This script contains helper methods for the model, eg: computing losses, forward pass in an epoch


'''

import torch
import numpy as np
import time

def l2_normalize_batch(x):
    return x.div( x.norm(p=2, dim=1, keepdim=True) )


# use this to compute objective function, where the target distribution has 1/X probability for each of the X words in the input
def soft_cross_entropy(input, target):
    return torch.mean(torch.sum(-target * torch.nn.functional.log_softmax(input, dim=1), dim=1))

def compute_triplet_loss(anchor, positive, negative, margin):
    b_size, dim_size = anchor.size()

    anchor_normalized = l2_normalize_batch(anchor)
    positive_normalized = l2_normalize_batch(positive)
    negative_normalized = l2_normalize_batch(negative)


    positive_dot = torch.bmm(anchor_normalized.view(b_size, 1, dim_size), positive_normalized.view(b_size, dim_size, 1))
    negative_dot = torch.bmm(anchor_normalized.view(b_size, 1, dim_size), negative_normalized.view(b_size, dim_size, 1))

    losses = torch.nn.functional.relu(1.0 + negative_dot - positive_dot)
    return losses.mean()




def compute_world_loss(ignore_none_world, none_id, world_logits, world_targets_t):
    if ignore_none_world:
        loss = torch.nn.CrossEntropyLoss(ignore_index=none_id)
    else:
        loss = torch.nn.CrossEntropyLoss()
    return loss(world_logits, world_targets_t)


def compute_world_pred_acc(pred_logits, targets_t, do_ignore_index, ignore_index):
    world_pred = torch.argmax(pred_logits, 1).detach().cpu().numpy()
    labels = targets_t.detach().cpu().numpy()
    if do_ignore_index:
        valid = labels != ignore_index
        correct = world_pred == labels
        correct_and_valid = correct * valid
        return np.sum(correct_and_valid) / np.sum(valid)
    return (np.mean(world_pred == labels))



def run_epoch(net, optim, batch_intervals_train, input_vector_world_id_list, input_vector_list_neg, args, train):
    device = args.device
    triplet_loss_margin = args.triplet_loss_margin
    triplet_loss_weight = args.triplet_loss_weight
    world_clas_weight = args.world_clas_weight
    ortho_weight = args.ortho_weight

    ep_loss = 0.
    ep_tri_loss = 0.
    ep_re_loss = 0.
    ep_or_loss = 0.
    ep_world_class_loss = 0.

    start_time = time.time()
    net.train()


    for b_idx, (start, end) in enumerate(batch_intervals_train):
        # print(start, end)

        batch_data = input_vector_world_id_list[start:end]
        batch_input_vec = [vec_id_pair[0] for vec_id_pair in batch_data]
        batch_data_t = torch.FloatTensor(np.array(batch_input_vec)).to(device)

        batch_data_neg = input_vector_list_neg[start:end]
        batch_data_neg_t = torch.FloatTensor(np.array(batch_data_neg)).to(device)

        recomb, world_logits = net(batch_data_t)


        triplet_loss = triplet_loss_weight * compute_triplet_loss(recomb, batch_data_t, batch_data_neg_t, triplet_loss_margin)

        # construct world classification target
        world_targets = [vec_id_pair[1] for vec_id_pair in batch_data]
        world_targets_t = torch.LongTensor(np.array(world_targets)).to(device)

        # compute world loss using the doc here: https://pytorch.org/docs/stable/nn.html#crossentropyloss
        world_class_loss = world_clas_weight * compute_world_loss(args.ignore_none_world, args.none_id, world_logits, world_targets_t)

        # compute orthogonality penalty on dictionary
        X = torch.nn.functional.normalize(net.X, dim=0)
        ortho_loss = ortho_weight * torch.sum((torch.mm(X, X.t()) - \
                                               torch.eye(X.size()[0]).to(device)) ** 2)

        batch_loss = triplet_loss + ortho_loss + world_class_loss

        if train: # at training time we perform gradient updates
            batch_loss.backward()
            optim.step()
            optim.zero_grad()

        # else: # at validation time we compute prediction accuracies
        pred_acc = compute_world_pred_acc(world_logits, world_targets_t, args.ignore_none_world, args.none_id)



        ep_loss += batch_loss.item()
        ep_tri_loss += triplet_loss.item()
        ep_or_loss += ortho_loss.item()
        ep_world_class_loss += world_class_loss.item()

    ep_loss = ep_loss / len(batch_intervals_train)
    ep_tri_loss = ep_tri_loss / len(batch_intervals_train)
    ep_or_loss = ep_or_loss / len(batch_intervals_train)
    ep_world_class_loss = ep_world_class_loss / len(batch_intervals_train)

    signature = 'TRAIN' if train == True else 'VALID'

    ep_info = '[%s] loss: %0.4f, %0.4f, %0.2f, %0.4f (all, tri, wo, or), accuracy:%0.5f, time: %0.2f s' % (signature,
        ep_loss, ep_tri_loss, ep_world_class_loss, ep_or_loss, pred_acc, time.time() - start_time)

    print(ep_info)

