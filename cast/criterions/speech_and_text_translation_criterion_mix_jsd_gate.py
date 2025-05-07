# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig, 
    label_smoothed_nll_loss,
)
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch.nn.functional as F


@dataclass
class SpeechAndTextTranslationCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    mt_finetune: bool = field(
        default=False,
        metadata={"help": "st + mt multi-task finetune"},
    )
    mix_ratio: float = field(
        default=0.0,
        metadata={"help": "mix retio"},
    )
    jsd_weight: float = field(
        default=0.0,
        metadata={"help": "jsd weight"},
    )    
    gate_weight: float = field(
        default=0.0,
        metadata={"help": "jsd weight"},
    )

@register_criterion(
    "speech_and_text_translation_mix_jsd_gate", dataclass=SpeechAndTextTranslationCriterionConfig
)
class SpeechAndTextTranslationCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        mt_finetune=False,
        mix_ratio=0.0,
        jsd_weight=0.0,
        gate_weight=0.0
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.pad_idx = task.target_dictionary.pad()
        self.mt_finetune = mt_finetune
        self.mix_ratio = mix_ratio
        self.jsd_weight = jsd_weight
        self.gate_weight = gate_weight

    def compute_loss_with_lprobs(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss, lprobs, target

    def forward_st(self, model, sample, reduce):
        audio_input = {
            "src_tokens": sample["net_input"]["audio"],
            "src_lengths": sample["net_input"]["audio_lengths"],
            "mode": "st",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
            "start": sample["start"],
        }
        audio_output = model(**audio_input)
        loss, _ = self.compute_loss(model, audio_output, sample, reduce=reduce)
        return loss

    def forward_st_with_x(self, model, sample, audio_x, audio_encoder_padding_mask, reduce):
        audio_input = {
            "src_tokens": sample["net_input"]["audio"],
            "src_lengths": sample["net_input"]["audio_lengths"],
            "mode": "st",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
            "start": sample["start"],
            "x": audio_x,
            "encoder_padding_mask": audio_encoder_padding_mask,
        }
        audio_output = model(**audio_input)
        loss, _, lprobs, target = self.compute_loss_with_lprobs(model, audio_output, sample, reduce=reduce)
        gate = audio_output[1]["encoder_out"]["gate"].transpose(0, 1).squeeze(-1)   # T x B x 1 -> B x T x 1 -> B x T
        return loss, lprobs, target, gate

    def forward_mt(self, model, sample, reduce):
        text_input = {
            "src_tokens": sample["net_input"]["source"],
            "src_lengths": sample["net_input"]["source_lengths"],
            "mode": "mt",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
        }
        text_output = model(**text_input)
        loss, _ = self.compute_loss(model, text_output, sample, reduce=reduce)
        return loss

    def forward_mt_with_x(self, model, sample, text_x, text_encoder_padding_mask, reduce):
        text_input = {
            "src_tokens": sample["net_input"]["source"],
            "src_lengths": sample["net_input"]["source_lengths"],
            "mode": "mt",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
            "start": sample["start"],
            "x": text_x,
            "encoder_padding_mask": text_encoder_padding_mask,
        }
        text_output = model(**text_input)
        loss, _, lprobs, target = self.compute_loss_with_lprobs(model, text_output, sample, reduce=reduce)
        return loss, lprobs, target

    def forward_ext_mt(self, model, sample, reduce):
        text_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, text_output, sample, reduce=reduce)
        return loss

    def compute_jsd_loss(self, lprobs_st, lprobs_mt, target_st, target_mt, ignore_index):
        kl_loss_st = F.kl_div(lprobs_mt, lprobs_st, log_target=True, reduction="none").sum(-1)
        kl_loss_mt = F.kl_div(lprobs_st, lprobs_mt, log_target=True, reduction="none").sum(-1)
        pad_mask = target_st.eq(ignore_index)
        kl_loss_st.masked_fill_(pad_mask, 0.0)
        pad_mask = target_mt.eq(ignore_index)
        kl_loss_mt.masked_fill_(pad_mask, 0.0)
        kl_loss_st = kl_loss_st.sum()
        kl_loss_mt = kl_loss_mt.sum()
        kl_loss = (kl_loss_st + kl_loss_mt) / 2.0
        return kl_loss

    def compute_gate_loss(self, model, sample, audio_x, audio_encoder_padding_mask, c_lprobs, target, gate):
        audio_input = {
            "src_tokens": sample["net_input"]["audio"],
            "src_lengths": sample["net_input"]["audio_lengths"],
            "mode": "st",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
            "start": sample["start"],
            "x": audio_x,
            "encoder_padding_mask": audio_encoder_padding_mask,
            "is_single_input": True,
        }
        audio_output = model(**audio_input)
        s_lprobs, _ = self.get_lprobs_and_target(model, audio_output, sample)
        if target.dim() == c_lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        c_lprobs = c_lprobs.gather(dim=-1, index=target)    # B x T x 1
        s_lprobs = s_lprobs.gather(dim=-1, index=target)

        pad_mask = target.eq(self.pad_idx)    # B x T x 1
        c_lprobs = c_lprobs.masked_fill(pad_mask, 0.0)
        s_lprobs = s_lprobs.masked_fill(pad_mask, 0.0)

        pad_mask = pad_mask.squeeze(-1)     # B x T
        c_lprobs = c_lprobs.squeeze(-1)     # B x T
        s_lprobs = s_lprobs.squeeze(-1)

        c_sum_probs = c_lprobs.sum(-1)
        s_sum_probs = s_lprobs.sum(-1)

        label = (c_sum_probs >= s_sum_probs).int()   # 0: sent-level is better; 1: doc-level is better

        gate = torch.where(gate > 0.001, gate, 0.001)   # makesure gate > 0.001 to avoid inf in training
        gate = torch.log(gate)

        # eq：x = x_local * g + x_global * (1 - g)
        label_mask = torch.zeros_like(gate)
        label_mask[:,] = label.unsqueeze(-1)    # B x T
        gate = gate.masked_fill(audio_encoder_padding_mask, 0.0)    # B x T
        gate = gate.masked_fill(label_mask.bool(), 0.0)   # B x T
        gate_loss = -self.gate_weight * gate.sum()
        return gate_loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        st_loss, mt_loss, ext_mt_loss = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()
        st_size, mt_size, ext_mt_size = 0, 0, 0
        output_jsd = torch.Tensor([0]).cuda()
        gate_loss = torch.Tensor([0]).cuda()

        mode = sample["net_input"]["mode"]
        if mode == "st":
            if self.mt_finetune and self.training:
                
                audio_x, audio_encoder_padding_mask, input_lengths, audio_emb = model.encoder.forward_audio(
                    sample["net_input"]["audio"], sample["net_input"]["audio_lengths"]
                )
                st_loss, lprobs_st, target_st, gate = self.forward_st_with_x(model, sample, audio_x, audio_encoder_padding_mask, reduce)
                
                text_x, text_encoder_padding_mask, text_emb = model.encoder.forward_embedding(sample["net_input"]["source"])
                if self.mix_ratio > 0.0:
                    text_x, text_encoder_padding_mask = model.encoder.forward_mix(
                        audio_emb, input_lengths, text_emb, sample["net_input"]["source_lengths"], sample["start"], sample["extind"], self.mix_ratio
                    )   # B x T x C
                mt_loss, lprobs_mt, target_mt = self.forward_mt_with_x(model, sample, text_x, text_encoder_padding_mask, reduce)

                if self.jsd_weight > 0.0:
                    output_jsd = self.compute_jsd_loss(lprobs_st, lprobs_mt, target_st, target_mt, self.padding_idx)

                if self.gate_weight > 0.0:
                    gate_loss = self.compute_gate_loss(model, sample, audio_x, audio_encoder_padding_mask, lprobs_st, target_st, gate)

                loss = st_loss + mt_loss + self.jsd_weight * output_jsd + gate_loss
                st_size = mt_size = sample_size = sample["ntokens"]
            else:   # inference
                loss = st_loss = self.forward_st(model, sample, reduce)
                st_size = sample_size = sample["ntokens"]
        elif mode == "ext_mt":
            loss = ext_mt_loss = self.forward_ext_mt(model, sample, reduce)
            ext_mt_size = sample_size = sample["ntokens"]

        logging_output = {
            "loss": loss.data,
            "st_loss": st_loss.data,
            "st_sample_size": st_size,
            "mt_loss": mt_loss.data,
            "mt_sample_size": mt_size,
            "jsd_loss": output_jsd, 
            "gate_loss": gate_loss, 
            "ext_mt_loss": ext_mt_loss.data,
            "ext_mt_sample_size": ext_mt_size,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        st_loss_sum = sum(log.get("st_loss", 0) for log in logging_outputs)
        mt_loss_sum = sum(log.get("mt_loss", 0) for log in logging_outputs)
        jsd_loss_sum = sum(log.get("jsd_loss", 0) for log in logging_outputs)
        gate_loss_sum = sum(log.get("gate_loss", 0) for log in logging_outputs)
        ext_mt_loss_sum = sum(log.get("ext_mt_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        st_sample_size = sum(log.get("st_sample_size", 0) for log in logging_outputs)
        mt_sample_size = sum(log.get("mt_sample_size", 0) for log in logging_outputs)
        ext_mt_sample_size = sum(log.get("ext_mt_sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "st_loss", st_loss_sum / st_sample_size / math.log(2) if st_sample_size != 0 else 0, st_sample_size, round=3
        )
        metrics.log_scalar(
            "mt_loss", mt_loss_sum / mt_sample_size / math.log(2) if mt_sample_size != 0 else 0, mt_sample_size, round=3
        )
        metrics.log_scalar(
            "jsd_loss", jsd_loss_sum / mt_sample_size / math.log(2) if mt_sample_size != 0 else 0, mt_sample_size, round=3
        )
        metrics.log_scalar(
            "gate_loss", gate_loss_sum / len(logging_outputs) / math.log(2) if mt_sample_size != 0 else 0, mt_sample_size, round=3
        )
        metrics.log_scalar(
            "ext_mt_loss", ext_mt_loss_sum / ext_mt_sample_size / math.log(2) if ext_mt_sample_size != 0 else 0, ext_mt_sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True