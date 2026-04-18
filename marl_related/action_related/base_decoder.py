class BaseActionDecoder:
    def get_action_dim(self, num_neighbors):
        """Define the output dimension for the neural network"""
        raise NotImplementedError
        
    def decode(self, state, raw_actions, num_edge, neighbors_map):
        """Rescale raw actions from [0, 1] to the environment's physical action range"""
        raise NotImplementedError