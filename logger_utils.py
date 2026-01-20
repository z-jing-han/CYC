import os
import logging
import numpy as np
from config import Config

class SimulationLogger:
    def __init__(self, algorithm_name, input_filename):
        self.algorithm_name = algorithm_name
        self.base_name = os.path.splitext(os.path.basename(input_filename))[0]
        
        self.log_dir = 'logs'
        self.output_dir = 'outputs'
        self.detailed_dir = 'detailed_stats'
        
        for directory in [self.log_dir, self.output_dir, self.detailed_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.log_file = os.path.join(self.log_dir, f'log_{self.algorithm_name}_{self.base_name}.txt')
        self.stats_file = os.path.join(self.detailed_dir, f'stats_{self.algorithm_name}_{self.base_name}.csv')
        
        # Set Logging
        self.logger = logging.getLogger(f"SimLogger_{algorithm_name}")
        self.logger.setLevel(logging.INFO)
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        file_handler = logging.FileHandler(self.log_file, mode='w')
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Init CSV header
        with open(self.stats_file, 'w') as f:
            header = "TimeSlot,"
            # Edge Headers
            for i in range(Config.NUM_EDGE_SERVERS):
                p = f"Edge{i+1}"
                header += (f"{p}_Arrival(bits),{p}_Q_Pre(bits),{p}_Q_Post(bits),"
                           f"{p}_Proc_Local(bits),{p}_Tx_Peer(bits),{p}_Tx_Cloud(bits),"
                           f"{p}_Energy_Comp(J),{p}_Energy_Tx(J),{p}_Carbon(g),")
            
            # Cloud Headers
            for i in range(Config.NUM_CLOUD_SERVERS):
                p = f"Cloud{i+1}"
                header += (f"{p}_Q_Pre(bits),{p}_Q_Post(bits),"
                           f"{p}_Proc(bits),{p}_Rx_Edge(bits),"
                           f"{p}_Energy_Comp(J),{p}_Carbon(g),")
            
            header += "Total_Carbon(g),Avg_System_Q(bits)\n"
            f.write(header)
            
        print(f"Logger initialized. Logs at: {self.log_file}, Stats at: {self.stats_file}")

    def log_step(self, metrics):
        self._write_csv_stats(metrics)
        self._write_text_log(metrics)

    def _write_csv_stats(self, metrics):
        step = metrics['time_step']
        edge_data = metrics['edge_metrics']
        cloud_data = metrics['cloud_metrics']
        glob = metrics['global_metrics']
        
        with open(self.stats_file, 'a') as f:
            line = f"{step},"
            
            for i in range(Config.NUM_EDGE_SERVERS):
                e = edge_data[i]
                line += (f"{e['arrival']:.2f},{e['q_pre']:.2f},{e['q_post']:.2f},"
                         f"{e['proc_local']:.2f},{e['tx_peer']:.2f},{e['tx_cloud']:.2f},"
                         f"{e['energy_comp']:.4e},{e['energy_tx']:.4e},{e['carbon']:.6f},")
            
            for i in range(Config.NUM_CLOUD_SERVERS):
                c = cloud_data[i]
                line += (f"{c['q_pre']:.2f},{c['q_post']:.2f},"
                         f"{c['proc']:.2f},{c['rx_edge']:.2f},"
                         f"{c['energy_comp']:.4e},{c['carbon']:.6f},")
                
            line += f"{glob['total_carbon']:.6f},{glob['avg_q']:.2f}\n"
            f.write(line)

    def _write_text_log(self, metrics):
        log = self.logger.info
        step = metrics['time_step']
        edge_data = metrics['edge_metrics']
        cloud_data = metrics['cloud_metrics']
        glob = metrics['global_metrics']
        
        log(f"---------------")
        log(f"Time Slot {step}:")
        
        TO_MB = 1.0 / Config.MB_TO_BITS
        
        for i in range(Config.NUM_EDGE_SERVERS):
            e = edge_data[i]
            log(f"Edge Server {i+1}:")
            log(f"  Arrival: {e['arrival']*TO_MB:.4f} MB")
            log(f"  Start Queue: {e['q_pre']*TO_MB:.4f} MB -> End Queue: {e['q_post']*TO_MB:.4f} MB")
            log(f"  Processed (Local): {e['proc_local']*TO_MB:.4f} MB")
            log(f"  Offloaded -> Peer: {e['tx_peer']*TO_MB:.4f} MB, Cloud: {e['tx_cloud']*TO_MB:.4f} MB")
            log(f"  Energy: Comp={e['energy_comp']:.4f} J, Tx={e['energy_tx']:.4f} J")
            log(f"  Carbon: {e['carbon']:.6f} g")

        for i in range(Config.NUM_CLOUD_SERVERS):
            c = cloud_data[i]
            log(f"Cloud Server {i+1}:")
            log(f"  Start Queue: {c['q_pre']*TO_MB:.4f} MB -> End Queue: {c['q_post']*TO_MB:.4f} MB")
            log(f"  Processed: {c['proc']*TO_MB:.4f} MB")
            log(f"  Rx from Edge: {c['rx_edge']*TO_MB:.4f} MB")
            log(f"  Energy: {c['energy_comp']:.4f} J, Carbon: {c['carbon']:.6f} g")
            
        log(f"Total Carbon Emission: {glob['total_carbon']:.6f} g")
        log(f"Avg System Queue: {glob['avg_q']*TO_MB:.4f} MB")

    def close(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)