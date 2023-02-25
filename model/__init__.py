__all__ = ["Net", "dual_transf", "single_trans"]

# Files in this folder were copied from the github repository https://github.com/key2miao/TSTNN
# based on https://arxiv.org/abs/2103.09963. 
# I did some tweaks to make the code more readable and more generalized.
# Also changed the whole principle on how the model works. You input a noisy spectrogram into it
# and model outputs noise reduced spectrogram