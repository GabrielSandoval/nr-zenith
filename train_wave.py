import sys
from utils import *

character = sys.argv[1]
print("Testing Lair: {}".format(character))
wave_model, wave_scaler = train_wave(character)