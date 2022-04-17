import math
import torch
import sys, pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin



class DepthEstimate():

    def __init__(self, depth_pred, num_bins, max_depth):
        # depth_pred: BxCxHxW tensor of depth probabilities over C depth bins
        # Convert logprobabilities to probabilities
        if depth_pred.max() < 0:
            depth_pred = torch.exp(depth_pred)
        assert ((depth_pred.sum(dim=1) - 1).abs() < 1e-3).all(), "Depth prediction needs to be a simplex at each pixel"

        self.depth_pred = depth_pred
        self.num_bins = num_bins
        self.max_depth = max_depth

    def to(self, device):
        depth_pred = self.depth_pred.to(device)
        return DepthEstimate(depth_pred, self.num_bins, self.max_depth)

    def domain(self, res=None):
        if res is None:
            res = self.num_bins
        return torch.arange(0, res, 1, device=self.depth_pred.device)[None, :, None, None]

    def domain_image(self, res=None):
        if res is None:
            res = self.num_bins
        domain = self.domain(res)
        domain_image = domain.repeat((1, 1, self.depth_pred.shape[2], self.depth_pred.shape[3])) * (self.max_depth / res)
        return domain_image

    def mle(self):
        mle_depth = self.depth_pred.argmax(dim=1, keepdim=True).float() * (self.max_depth / self.num_bins)
        return mle_depth

    def expectation(self):
        expected_depth = (self.domain() * self.depth_pred).sum(dim=1, keepdims=True) * (self.max_depth / self.num_bins)
        return expected_depth

    def spread(self):
        spread = (self.mle() - self.expectation()).abs()
        return spread

    def percentile(self, which):
        domain = self.domain()
        cumsum = self.depth_pred.cumsum(dim=1)
        pctlbin = ((cumsum < which) * domain).max(dim=1, keepdim=True).values
        pctldepth = pctlbin * (self.max_depth / self.num_bins)
        return pctldepth

    def get_trustworthy_depth(self, include_mask=None, confidence=0.9, max_conf_int_width_prop=0.30):
        conf_int_lower = self.percentile((1 - confidence) / 2)
        conf_int_upper = self.percentile(1 - (1 - confidence) / 2)

        spread = conf_int_upper - conf_int_lower
        max_conf_int_width = self.expectation() * max_conf_int_width_prop
        trusted_mask = spread < max_conf_int_width

        accept_mask = trusted_mask

        if include_mask is not None:
            # Apply looser criteria for objects that the agent is actively looking for
            include_mask_criteria = spread < self.expectation()
            #ForkedPdb().set_trace()
            device = include_mask.device
            include_mask_criteria = include_mask_criteria.to(device)
            include_mask_solid = include_mask_criteria * include_mask
            accept_mask = accept_mask.to(device)
            accept_mask = accept_mask.bool() + include_mask_solid.bool()

        est_depth = self.mle()
        est_depth = est_depth.to("cpu")
        accept_mask = accept_mask.to("cpu")
        trustworthy_depth = est_depth * accept_mask
        return trustworthy_depth
