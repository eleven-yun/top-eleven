from torch import nn

# TODO(diwei): Remove this functinon
def count_parameters(model):
    return sum(par.numel() for par in model.parameters() if par.requires_grad)


# TODO(diwei): Remove this functinon
def initialize_weights(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.kaiming_uniform_(model.weight.data)


# TODO(diwei): Remove this functinon
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
