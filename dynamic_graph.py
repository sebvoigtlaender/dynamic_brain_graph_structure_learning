

def split(x):
    x = pt.rand(cfg.batch_size, cfg.n_neurons, cfg.T)
    T = (cfg.T - 2*(cfg.len_window - 1) - 1)//(cfg.stride+1)
    x_cprs = pt.stack([x[:, :, t*cfg.stride:t*cfg.stride+cfg.len_window] for t in range(T)], -1)
    x_cprs = pt.transpose(x_cprs, 1, 2)
    return x_cprs

def ebd_brain_region():

class Core(pt.nn.Module):

    '''
    Basic multilayer perceptron
    '''

    def __init__(self,
                 n_input_neurons: int,
                 n_hidden_neurons: int,
                 n_output_neurons: int, 
                 n_layers: int,
                 act_fn: Optional[str] = 'relu',
                 bias: Optional[bool] = True) -> None:

        super().__init__()
        self.act_fn = act_fn
        self.input_layer = pt.nn.Linear(n_input_neurons, n_hidden_neurons, bias = bias)
        self.hidden_layers = pt.nn.ModuleList([pt.nn.Linear(n_hidden_neurons, n_hidden_neurons, bias = bias) for i in range(n_layers)])
        self.output_layer = pt.nn.Linear(n_hidden_neurons, n_output_neurons, bias = bias)
        
    def forward(self, x: pt.Tensor) -> pt.Tensor:
        x = self.input_layer(x)
        x = activation_fn(self.act_fn)(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = activation_fn(self.act_fn)(x)
        x = self.output_layer(x)
        return x




def spatial_attention():

def temporal_attention():

def graph_contruction():

def dynamic_graph_net():
    x = split(x)
