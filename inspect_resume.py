from pipeline.session_index import index_sessions
from pipeline_gpt2class import _checkpoint_exists

input_path = r'\\wsl.localhost\Ubuntu-24.04\home\peppe\tuh_eeg_seizure_v2.0.6'
sessions = index_sessions(input_path)
print('sessions', len(sessions))
for key in list(sessions.keys())[:20]:
    print(key, 'raw_exists', _checkpoint_exists('checkpoints', key, 'raw'), 'proc_exists', _checkpoint_exists('checkpoints', key, 'proc'))
