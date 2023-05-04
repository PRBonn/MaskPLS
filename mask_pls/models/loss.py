# Modified by Rodrigo Marcuzzi from https://github.com/facebookresearch/Mask2Former
from itertools import filterfalse

import torch
import torch.nn.functional as F
from mask_pls.models.matcher import HungarianMatcher
from mask_pls.utils.misc import (get_world_size, is_dist_avail_and_initialized,
                                 pad_stack, sample_points)
from torch import nn
from torch.autograd import Variable


class MaskLoss(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, cfg, data_cfg):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = data_cfg.NUM_CLASSES
        self.ignore = data_cfg.IGNORE_LABEL
        self.matcher = HungarianMatcher(cfg.WEIGHTS, cfg.P_RATIO)

        self.weight_dict = {
            cfg.WEIGHTS_KEYS[i]: cfg.WEIGHTS[i] for i in range(len(cfg.WEIGHTS))
        }

        self.eos_coef = cfg.EOS_COEF

        weights = torch.ones(self.num_classes + 1)
        weights[0] = 0.0
        weights[-1] = self.eos_coef
        self.weights = weights

        # pointwise mask loss parameters
        self.num_points = cfg.NUM_POINTS
        self.n_mask_pts = cfg.NUM_MASK_PTS

    def forward(self, outputs, targets, masks_ids, coors):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors:
                 pred_logits: [B,Q,NClass+1]
                 pred_masks: [B,Pts,Q]
                 aux_outputs: list of dicts ['pred_logits'], ['pred_masks'] for each
                              intermediate output
             targets: dict of lists len BS:
                          classes: semantic class for each GT mask
                          masks: [N_Masks, Pts] binary masks for each point
        """
        losses = {}
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        targ = targets
        num_masks = sum(len(t) for t in targ["classes"])
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        outputs_no_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_no_aux, targ)

        losses.update(
            self.get_losses(outputs, targ, indices, num_masks, masks_ids, coors)
        )
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targ)
                l_dict = self.get_losses(
                    aux_outputs, targ, indices, num_masks, masks_ids, coors
                )
                l_dict = {f"{i}_" + k: v for k, v in l_dict.items()}
                losses.update(l_dict)

        losses = {
            l: losses[l] * self.weight_dict[k]
            for l in losses
            for k in self.weight_dict
            if k in l
        }

        return losses

    def get_losses(
        self, outputs, targets, indices, num_masks, masks_ids, coors, do_cls=True
    ):
        if do_cls:
            classes = self.loss_classes(outputs, targets, indices)
        else:
            classes = {}
        masks = self.loss_masks(outputs, targets, indices, num_masks, masks_ids)
        classes.update(masks)
        return classes

    def loss_classes(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "classes" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        pred_logits = outputs["pred_logits"].float()

        idx = self._get_pred_permutation_idx(indices)

        target_classes_o = torch.cat(
            [t[J] for t, (_, J) in zip(targets["classes"], indices)]
        ).to(pred_logits.device)

        target_classes = torch.full(
            pred_logits.shape[:2],
            self.num_classes,  # fill with class no_obj
            dtype=torch.int64,
            device=pred_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),
            target_classes,
            self.weights.to(pred_logits),
            ignore_index=self.ignore,
        )
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks, masks_ids):
        """Compute the losses related to the masks: the focal loss and the dice loss."""
        assert "pred_masks" in outputs

        masks = [t for t in targets["masks"]]
        n_masks = [m.shape[0] for m in masks]
        target_masks = pad_stack(masks)

        pred_idx = self._get_pred_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices, n_masks)
        pred_masks = outputs["pred_masks"]
        pred_masks = pred_masks[pred_idx[0], :, pred_idx[1]]
        target_masks = target_masks.to(pred_masks)
        target_masks = target_masks[tgt_idx]

        with torch.no_grad():
            idx = sample_points(masks, masks_ids, self.n_mask_pts, self.num_points)
            n_masks.insert(0, 0)
            nm = torch.cumsum(torch.tensor(n_masks), 0)
            point_labels = torch.cat(
                [target_masks[nm[i] : nm[i + 1]][:, p] for i, p in enumerate(idx)]
            )
        point_logits = torch.cat(
            [pred_masks[nm[i] : nm[i + 1]][:, p] for i, p in enumerate(idx)]
        )

        del pred_masks
        del target_masks

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        return losses

    def _get_pred_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices, n_masks):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        # From [B,id] to [id] of stacked masks
        cont_id = torch.cat([torch.arange(n) for n in n_masks])
        b_id = torch.stack((batch_idx, cont_id), axis=1)
        map_m = torch.zeros((torch.max(batch_idx) + 1, max(n_masks)))
        for i in range(len(b_id)):
            map_m[b_id[i, 0], b_id[i, 1]] = i
        stack_ids = [
            int(map_m[batch_idx[i], tgt_idx[i]]) for i in range(len(batch_idx))
        ]
        return stack_ids


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    # loss = 1 - (numerator + 0.001) / (denominator + 0.001)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule

################################################################################


class SemLoss(nn.Module):
    def __init__(self, w):
        super().__init__()

        self.ce_w, self.lov_w = w
        self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, outputs, targets):
        ce = self.cross_entropy(outputs, targets)
        lovasz = self.lovasz_softmax(F.softmax(outputs, dim=1), targets)
        loss = {"sem_ce": self.ce_w * ce, "sem_lov": self.lov_w * lovasz}
        return loss

    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def lovasz_softmax(self, probas, labels, classes="present", ignore=None):
        """
        Multi-class Lovasz-Softmax loss
          probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                  Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
          labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
          per_image: compute the loss per image instead of per batch
          ignore: void class labels
        """
        loss = self.lovasz_softmax_flat(
            *self.flatten_probas(probas, labels, ignore), classes=classes
        )
        return loss

    def lovasz_softmax_flat(self, probas, labels, classes="present"):
        """
        Multi-class Lovasz-Softmax loss
          probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
          labels: [P] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.0
        C = probas.size(1)
        losses = []
        class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
        for c in class_to_sum:
            fg = (labels == c).float()  # foreground for class c
            if classes == "present" and fg.sum() == 0:
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError("Sigmoid output possible only with 1 class")
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (Variable(fg) - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(
                torch.dot(errors_sorted, Variable(self.lovasz_grad(fg_sorted)))
            )
        return self.mean(losses)

    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        # Probabilities from SparseTensor.features already flattened
        N, C = probas.size()
        probas = probas.contiguous().view(-1, C)
        labels = labels.view(-1)
        if ignore is None:
            return probas, labels
        valid = labels != ignore
        vprobas = probas[torch.nonzero(valid).squeeze()]
        vlabels = labels[valid]
        return vprobas, vlabels

    def isnan(self, x):
        return x != x

    def mean(self, l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """
        l = iter(l)
        if ignore_nan:
            l = filterfalse(self.isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == "raise":
                raise ValueError("Empty mean")
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n
