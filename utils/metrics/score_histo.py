import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class ScoreHistogram(Metric):

    def __init__(self, s_range:list[int]=[0,1], s_bins:int=50, density: bool=False, common_norm: bool=False, **kwargs):
        super().__init__(**kwargs)

        self.s_range = s_range
        self.s_bins = s_bins
        self.density = density
        self._step_size = (s_range[1] - s_range[0]) / s_bins
        self.common_norm = common_norm

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")


    def update(self, preds: torch.Tensor, target: torch.Tensor, *args, **kwargs):

        self.preds.append(preds.flatten())
        self.target.append(target.flatten())
    
    def compute(self):

        preds = dim_zero_cat(self.preds).cpu()
        target = dim_zero_cat(self.target).cpu()

        outputs = {"score_histogram": {}}
        weights = torch.ones_like(preds) 

        if self.common_norm:
            _, inverse_idx, count = torch.unique(target, return_inverse=True, return_counts=True)
            proportion = count / torch.sum(count)
            weights = proportion[inverse_idx]
        
        for label in torch.unique(target):
            if self.density:
                outputs['score_histogram'][label], bin_edges = torch.histogram(preds[target==label], bins=self.s_bins, range=self.s_range, density=self.density) 
                outputs["score_histogram"][label] *= weights[target==label].mean()
            else:
                outputs['score_histogram'][label], bin_edges = torch.histogram(preds[target==label], bins=self.s_bins, range=self.s_range)
            outputs['bin_edges'] = bin_edges
            
        return outputs
            


    
    # def update(self, score: torch.Tensor, label: None | torch.Tensor = None):
        
    #     histo = {}
    #     bin_edges = self._bin_edges
    #     score = score.cpu()
    #     label = label.cpu()
    #     if self.labels:
    #         for l in self.labels:
    #             h, b = torch.histogram(score[label == l], bins=self.s_bins, range=self.s_range)
    #             histo[l] = h
    #             bin_edges = b
    #             self._histo[l] += h

    #         self._bin_edges = bin_edges
    #         if self.density:
    #             if self.common_norm:
    #                 norm = torch.sum(torch.concat([h for h in histo.values()])) * self._step_size
    #                 histo = {l: h / norm for l, h in histo.items()}
    #             else:
    #                 histo = {l: h / torch.sum(h * self._step_size) for l, h in histo.items()}
    #     else:
    #         histo, bin_edges = torch.histogram(score, bins=self.s_bins, range=self.s_range)
    #         self._histo += histo
    #         self._bin_edges = bin_edges
    #         if self.density:
    #             histo /= (torch.sum(histo) * self._step_size)

    #     return histo, self._bin_edges

    # def compute(self):

    #     if self.density:
    #         if self.labels:
    #             if self.common_norm:
    #                 norm = torch.sum(torch.concat([h for h in self._histo.values()])) * self._step_size
    #                 self._histo = {l: h / norm for l, h in self._histo.items()}
    #             else:
    #                 self._histo = {l: h / torch.sum(h * self._step_size) for l, h in self._histo.items()}
    #         else:
    #             self._histo /= torch.sum(self._histo * self._step_size)

    #     return self._histo, self._bin_edges


        




