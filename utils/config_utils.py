
def validate_neuron_config(condition_config):
    assert 'neuron_condition_config' in condition_config, \
        "Neuron conditioning desired but neuron condition config missing"
    assert 'neuron_embed_dim' in condition_config['neuron_condition_config'], \
        "neuron_embed_dim missing in neuron condition config"
    

