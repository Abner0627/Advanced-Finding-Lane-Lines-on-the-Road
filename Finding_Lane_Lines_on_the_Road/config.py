import numpy as np

#======================
solidWhiteRight_src = np.float32(
            [[153, 540],  # Bottom left
            [430, 337],  # Top left
            [536, 337],  # Top right
            [872, 540]]) # Bottom right

solidWhiteRight_dst = np.float32(
            [[153, 540],  # Bottom left
            [153, 0],  # Top left
            [872, 0],  # Top right
            [872, 540]]) # Bottom right 
#======================
challenge_src = np.float32(
            [[280, 666],  # Bottom left
             [595, 460],  # Top left
             [735, 460],  # Top right
             [1080, 666]]) # Bottom right

challenge_dst = np.float32(
            [[280, 666],  # Bottom left
             [280, 0],  # Top left
             [1080, 0],  # Top right
             [1080, 666]]) # Bottom right
#======================
solidYellowLeft_src = np.float32(
            [[100, 540],  # Bottom left
             [430, 340],  # Top left
             [540, 340],  # Top right
             [840, 540]]) # Bottom right

solidYellowLeft_dst = np.float32(
            [[100, 540],  # Bottom left
             [100, 0],  # Top left
             [840, 0],  # Top right
             [840, 540]]) # Bottom right             