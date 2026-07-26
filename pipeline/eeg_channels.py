# 17 target EEG channels the pipeline extracts, in both conventions (AR,LE).
# Given edf file on has either present
CHANNELS_TO_INCLUDE = [
    'EEG T6-REF', 'EEG T5-REF', 'EEG T4-REF', 'EEG T3-REF',
    'EEG P4-REF', 'EEG P3-REF', 'EEG O2-REF', 'EEG O1-REF',
    'EEG FP2-REF', 'EEG FP1-REF', 'EEG F8-REF', 'EEG F7-REF',
    'EEG F4-REF', 'EEG F3-REF', 'EEG CZ-REF', 'EEG C4-REF',
    'EEG C3-REF', 'EEG T6-LE', 'EEG T5-LE', 'EEG T4-LE',
    'EEG T3-LE', 'EEG P4-LE', 'EEG P3-LE', 'EEG O2-LE',
    'EEG O1-LE', 'EEG FP2-LE', 'EEG FP1-LE', 'EEG F8-LE',
    'EEG F7-LE', 'EEG F4-LE', 'EEG F3-LE', 'EEG CZ-LE',
    'EEG C4-LE', 'EEG C3-LE',
]

N_TARGET_CHANNELS = 17