import haiku as hk

def build_dqn(num_hidden_units: int, num_actions: int) -> hk.Transformed:
    
    def q_network(observation):
        return hk.nets.MLP((num_hidden_units,num_hidden_units,num_actions))(observation)
    
    return hk.without_apply_rng(hk.transform(q_network))