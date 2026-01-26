import os
import logging
import numpy as np
from config import Config

class SimulationLogger:
    def __init__(self, algorithm_name, output_dir):
        """
        Init Logger
        :param algorithm_name: Algroithm (using in filename)
        :param output_dir: User defined output directory (e.g., Base_Output)
        """
        self.algorithm_name = algorithm_name
        
        self.output_root = output_dir
        self.log_dir = os.path.join(self.output_root, 'logs')
        self.csv_dir = os.path.join(self.output_root, 'csv')
        
        for directory in [self.log_dir, self.csv_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.log_file = os.path.join(self.log_dir, f'log_{self.algorithm_name}.txt')
        self.stats_file = os.path.join(self.csv_dir, f'stats_{self.algorithm_name}.csv')
        
        self.logger = logging.getLogger(f"SimLogger_{algorithm_name}")
        self.logger.setLevel(logging.INFO)
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            header = "TimeSlot,"
            for i in range(Config.NUM_EDGE_SERVERS):
                p = f"Edge{i+1}"
                header += (f"{p}_Arrival(bits),{p}_Q_Pre(bits),{p}_Q_Post(bits),"
                           f"{p}_Proc_Local(bits),{p}_Tx_Peer(bits),{p}_Tx_Cloud(bits),"
                           f"{p}_Energy_Comp(J),{p}_Energy_Tx(J),{p}_Carbon(g),")
            
            for i in range(Config.NUM_CLOUD_SERVERS):
                p = f"Cloud{i+1}"
                header += (f"{p}_Q_Pre(bits),{p}_Q_Post(bits),"
                           f"{p}_Proc(bits),{p}_Rx_Edge(bits),"
                           f"{p}_Energy_Comp(J),{p}_Carbon(g),")
            
            header += "Total_Carbon(g),Avg_System_Q(bits)\n"
            f.write(header)

    def log_step(self, metrics):
        self._write_csv_stats(metrics)
        self._write_text_log(metrics)

    def _write_csv_stats(self, metrics):
        step = metrics['time_step']
        edge_data = metrics['edge_metrics']
        cloud_data = metrics['cloud_metrics']
        glob = metrics['global_metrics']
        
        with open(self.stats_file, 'a', encoding='utf-8') as f:
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