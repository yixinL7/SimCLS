# modified from https://github.com/maszhongming/MatchSum
import torch
from torch import nn
from transformers import RobertaModel


def RankingLoss(score, summary_score=None, margin=0, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False):
    ones = torch.ones_like(score)
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(score, score, ones)
    # candidate loss
    n = score.size(1)
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss
    if no_gold:
        return TotalLoss
    # gold summary loss
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)
    return TotalLoss


class ReRanker(nn.Module):
    def __init__(self, encoder, pad_token_id):
        super(ReRanker, self).__init__()
        self.encoder = RobertaModel.from_pretrained(encoder)
        self.pad_token_id = pad_token_id

    def forward(self, text_id, candidate_id, summary_id=None, require_gold=True):
        
        batch_size = text_id.size(0)
        
        input_mask = text_id != self.pad_token_id
        out = self.encoder(text_id, attention_mask=input_mask)[0]
        doc_emb = out[:, 0, :]
        
        if require_gold:
            # get reference score
            input_mask = summary_id != self.pad_token_id
            out = self.encoder(summary_id, attention_mask=input_mask)[0]
            summary_emb = out[:, 0, :]
            summary_score = torch.cosine_similarity(summary_emb, doc_emb, dim=-1)

        candidate_num = candidate_id.size(1)
        candidate_id = candidate_id.view(-1, candidate_id.size(-1))
        input_mask = candidate_id != self.pad_token_id
        out = self.encoder(candidate_id, attention_mask=input_mask)[0]
        candidate_emb = out[:, 0, :].view(batch_size, candidate_num, -1)
        
        # get candidate score
        doc_emb = doc_emb.unsqueeze(1).expand_as(candidate_emb)
        score = torch.cosine_similarity(candidate_emb, doc_emb, dim=-1)

        output = {'score': score}
        if require_gold:
            output['summary_score'] = summary_score
        return output