from .base_decoder import BaseActionDecoder
from config import Config
import numpy as np

class FreqDecoder(BaseActionDecoder):
    def get_action_dim(self, num_neighbors=0):
        return 2 

    def decode(self, state, raw_actions, num_edge, neighbors_map=None):
        f_edge = np.zeros(num_edge)
        f_cloud = np.zeros(num_edge)
        Q_edge = state['Q_edge']
        Q_cloud = state['Q_cloud']
        
        for i in range(num_edge):
            action = raw_actions[i]
            
            needed_f_edge = (Q_edge[i] * Config.PHI) / Config.TIME_SLOT_DURATION
            max_valid_f_edge = min(Config.EDGE_F_MAX, needed_f_edge)
            
            needed_f_cloud = (Q_cloud[i] * Config.PHI) / Config.TIME_SLOT_DURATION
            max_valid_f_cloud = min(Config.CLOUD_F_MAX, needed_f_cloud)

            f_edge[i] = action[0] * max_valid_f_edge
            f_cloud[i] = action[1] * max_valid_f_cloud
        
        return {
            'f_edge': f_edge,
            'f_cloud': f_cloud
        }

class QueueDecoder(BaseActionDecoder):
    def get_action_dim(self, num_neighbors=0):
        return 2 

    def decode(self, state, raw_actions, num_edge, neighbors_map=None):
        f_edge = np.zeros(num_edge)
        f_cloud = np.zeros(num_edge)
        Q_edge = state['Q_edge']
        Q_cloud = state['Q_cloud']
        
        for i in range(num_edge):
            action = raw_actions[i]
            
            needed_f_edge = (Q_edge[i] * Config.PHI) / Config.TIME_SLOT_DURATION
            needed_f_cloud = (Q_cloud[i] * Config.PHI) / Config.TIME_SLOT_DURATION
            
            # action[0] is the process ratio
            # Can change to 1 - action[0] so that the action[0] will become ramain ratio
            process_ratio_edge = action[0]
            process_ratio_cloud = action[1]
            
            target_f_edge = process_ratio_edge * needed_f_edge
            f_edge[i] = min(Config.EDGE_F_MAX, target_f_edge)
            
            target_f_cloud = process_ratio_cloud * needed_f_cloud
            f_cloud[i] = min(Config.CLOUD_F_MAX, target_f_cloud)
        
        return {
            'f_edge': f_edge,
            'f_cloud': f_cloud
        }