from typing import Union

import torch
import torch.nn as nn

from lgp.models.teach.hlsm.hlsm_state_repr import TeachSpatialStateRepr
#from lgp.models.teach.hlsm.transformer_modules.transformer_layer import TransformerSideLayer
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


class StateReprEncoderPooled(nn.Module):

    def __init__(self, dmodel, nhead=8):
        super().__init__()
        self.task_layer = nn.Linear(dmodel, dmodel * nhead)
        self.state_layer = nn.Linear(TeachSpatialStateRepr.get_num_data_channels() * 2, dmodel)
        #self.enc_layer_3d = TransformerSideLayer(d_model=dmodel, n_head=1, dim_ff=dmodel, kvdim=dmodel)
        #self.enc_layer_1d = TransformerSideLayer(d_model=dmodel, n_head=1, dim_ff=dmodel, kvdim=dmodel)

        self.dmodel = dmodel
        self.nhead = nhead

    @classmethod
    def _make_pooled_repr(cls, state_reprs):
        b, c, w, l, h = state_reprs.data.data.shape
        #ForkedPdb().set_trace()
        #state_pooled = state_reprs.data.data.view([b, c, w*l*h]).max(dim=2).values
        state_pooled = torch.tensor(state_reprs.data.data.view([b, c, w*l*h]).numpy().max(axis=2))
        #ForkedPdb().set_trace()
        state_pooled_and_inv = torch.cat([state_pooled, state_reprs.inventory_vector], dim=1)
        return state_pooled_and_inv

    def forward(self,
                state_reprs: Union[TeachSpatialStateRepr, torch.tensor],
                task_embeddings: torch.tensor,
                action_hist_embeddings: torch.tensor
                ) -> torch.tensor:
        """
        Args:
            state_reprs: TeachSpatialStateRepr with data of shape BxCxWxLxH
            task_repr: TeachSpatialTaskRepr with data of shape BxD_{u}
            action_hist_repr: torch.tensor of shape BxD_{a}

        Returns:
            BxD_{s} dimensional batch of state representation embeddings.
        """
        #ForkedPdb().set_trace()
        state_reprs.data.data = state_reprs.data.data.to("cpu")
        if isinstance(state_reprs, TeachSpatialStateRepr):
            state_pooled_and_inv = StateReprEncoderPooled._make_pooled_repr(state_reprs)
        else:
            state_pooled_and_inv = state_reprs
        flat_repr = self.state_layer(state_pooled_and_inv.float())
        return flat_repr
