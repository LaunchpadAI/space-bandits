import torch
import torch.nn as nn
from collections import OrderedDict

class Wide_Deep(nn.Module):
    def __init__(self, wide_dim, deep_dim, action_dim, embeddings={}, deep_neurons=[32, 16], activation=nn.ReLU()):
        
        super(Wide_Deep, self).__init__()
        self.wide_dim = wide_dim
        self.deep_dim = deep_dim
        self.context_dim = wide_dim + deep_dim
        self.action_dim = action_dim
        self.deep_neurons = deep_neurons

        self.add_module('wide', nn.Module())
        self.add_module('deep', nn.Module())

        dims = {'wide':wide_dim, 'deep':deep_dim}
        for name, child_module in self.named_children():
            if name in embeddings.keys():
                assert dims[name] >= len(embeddings[name]), "Number of {} embedding features defined in embeddings should not more than {}_dim.".format(name, name)
                for embed in embeddings[name]:
                    child_module.add_module(embed[0], nn.Embedding(embed[1], embed[2]))
                    dims[name] += -1
                    dims[name] += embed[2]  
        self.dims = dims

        if self.dims['deep'] == 0:
            assert (self.dims['wide'] > 0), "Both wide_dim and deep_dim are 0, at least one of them needs a positive value." 
            print("This is a wide-only model.")
            self.z_dim = self.dims['wide']
        else:
            if self.dims['wide'] == 0:
                print("This is a deep-only model.")
            else:
                print("This is a wide and deep model.")
            assert len(deep_neurons) > 0, "deep_neurons must not be empty for the deep part."
            deep_layers = OrderedDict()
            layer_in = self.dims['deep']
            for i, layer_out in enumerate(deep_neurons):
                deep_layers['fc{}'.format(i)] = nn.Linear(layer_in, layer_out)
                deep_layers['activ{}'.format(i)] = activation
                layer_in = layer_out
            self.deep.add_module('NN', nn.Sequential(deep_layers))
            self.z_dim = self.dims['wide'] + deep_neurons[-1]

        self.lastlayer = nn.Linear(self.z_dim, self.action_dim)

    def get_z(self, x):
        inputs = {'wide':x[:, :self.wide_dim], 'deep':x[:, self.wide_dim:]}
        
        after_embed = {}
        for name, module in self.named_children():
            if name not in ['wide', 'deep']:
                break;
            i = 0
            for child in module.children():
                if 'Embedding' in torch.typename(child):
                    if name not in after_embed:
                        after_embed[name] = child(inputs[name][:, i].long())
                        i += 1
                    else:
                        after_embed[name] = torch.cat((after_embed[name], child(inputs[name][:, i].long())), dim=1)
                        i += 1 
            if name not in after_embed:
                after_embed[name] = inputs[name]
            else:
                after_embed[name] = torch.cat((after_embed[name], inputs[name][:, i:]), dim=1)

        if self.dims['deep'] == 0:
            z = after_embed['wide']
        elif self.dims['wide'] == 0:
            z = self.deep.NN(after_embed['deep'])
        else:
            z = torch.cat((after_embed['wide'], self.deep.NN(after_embed['deep'])), dim=1)
        
        return z

    def forward(self, x):
        z = self.get_z(x)
        out = self.lastlayer(z)
        return out
