import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open('demo_data/20250113_120238/Abalone/output_graph/PC_global_state.pkl', 'rb') as f:
    global_state = pickle.load(f)
print(global_state.logging.global_state_logging)
