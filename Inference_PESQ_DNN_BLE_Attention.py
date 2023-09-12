#####################################################################################
# Inference_PESQ_DNN_BLE_Attention:
# This script used to predict the coded speech PESQ score employing the non-intrusive PESQ-DNN model trained employing block-level embeddings (BLE) and attention.
# This model is trained with PESQNet Loss defined in (3). Details please check the paper. 
#
# Given data :
#       Coded speech files (.wav) in './Test_data/'
#       Trained PESQ-DNN model under './Trained_model/'
#       mean_for_Complex_Spectrum.mat (Mean value collect from UNNORMALIZED training coded speech spectrum for input, which is used to normalize input complex spectrum of PESQ-DNN to zero mean)
#       std_for_Complex_Spectrum.mat (Std value collect from UNNORMALIZED training coded speech spectrum for input, which is used to normalize input complex spectrum of PESQ-DNN to unit variance)
#
# Output data:
#       Print the mean value of the PESQ socres over all tested audio samples
#
# Functions:
#       audio_processing.py is used to perform FFT/IFFT for speech file processing.
#
# Technische Universität Braunschweig
# Institute for Communications Technology (IfN)
# Schleinitzstrasse 22
# 38106 Braunschweig
# Germany
# 2023 - 03 - 30
# (c) Ziyi Xu
#
# More technical details are introduced in the paper:
#   Z. Xu, M. Zhao and T. Fingscheidt, “Coded Speech Quality Measurement by a Non-Intrusive PESQ-DNN,” 
#   Submitted to IEEE/ACM T-ASLP
#
#
#####################################################################################



import os
import tensorflow as tf
import h5py
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Add, Multiply, Dot, Softmax, Average, Activation, Concatenate, LeakyReLU, Flatten, Permute, TimeDistributed, Bidirectional, Masking, ConvLSTM2D, Reshape, Lambda, Dense, LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling1D, AveragePooling1D, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import numpy as np
import scipy.io as sio
import scipy.io.wavfile as swave
import math
import hdf5storage as hdf5
from audio_processing import *
import soundfile as sf
from scipy import signal

#####################################################################################
# 1. Directory
#####################################################################################
print('> Loading data... ')

databank_root = './Test_data/' # give the directory where the test data .wav files are saved
file_extension = '.wav'

model_name = 'TRAINED_MODEL.h5' # give the path of the trained PESQ-DNN.

   
train_path_mean_info = './Trained_model/mean_for_Complex_Spectrum.mat' # give the path of the mean for normalization. This is a .mat file
train_path_std_info = './Trained_model/std_for_Complex_Spectrum.mat' # give the path of the std for normalization. This is a .mat file

#####################################################################################
# 2. Setteings
#####################################################################################
input_length = 260 
n1 = 32  #
n2 = 384  #

W_k1 = 5
W_k2 = 3

w1 = 1
w2 = 2
w3 = 4
w4 = 8

h_k = int(input_length/4)
w_k_1 = int(16/2)-w1+1
w_k_2 = int(16/2)-w2+1
w_k_3 = int(16/2)-w3+1
w_k_4 = int(16/2)-w4+1

LSTM_node=128

framing_params =   {'f_s': 16000,
                    'frame_length': 512,
                    'frame_shift': 256,
                    'K_fft': 512,
                    'K_mask': 260,
                    'feature_dim': 260,
                    'output_dim': 260,
                    'window': 'hann',
                    'sqrt_window_flag': False}
                    
frame_length = framing_params['frame_length']   # sample number
input_length = framing_params['K_mask']
fram_shift = framing_params['frame_shift']
K_fft = framing_params['K_fft']
Fs_Hz = framing_params['f_s']

if framing_params['window'] == 'hann':
    window = signal.hann(framing_params['frame_length'], sym=False)
elif framing_params['window'] == 'hamming':
    window = signal.hamming(framing_params['frame_length'], sym=False)
else:
    raise ValueError('The given window is not implemented. Check -window- parameter.')
#####################################################################################
# 3. Self-defined Functions
#####################################################################################

def k_squeez_dim(x):
    return K.squeeze(x, axis=-2)

def k_mean_layer(x):
    return K.mean(x, axis=1, keepdims=False)

def output_layer(x):
    k1=K.sigmoid(x)*3.6
    return k1 + 1.04
    
    
#####################################################################################
# 4. Load Data
#####################################################################################

print('  >> Loading training mean... ')
mat_mean_train = os.path.normcase(train_path_mean_info)
mean_infor_train = h5py.File(mat_mean_train, 'r')
train_mean_post = mean_infor_train.get('mean')
train_mean_post = np.array(train_mean_post)  # For converting to numpy array
train_mean_post = np.squeeze(train_mean_post)
print('mean shape: %s' % train_mean_post.shape[0])

print('  >> Loading training std... ')
mat_std_train = os.path.normcase(train_path_std_info)
std_infor_train = h5py.File(mat_std_train, 'r')
train_std_post = std_infor_train.get('std')
train_std_post = np.array(train_std_post)  # For converting to numpy array
train_std_post = np.squeeze(train_std_post)
print('std shape: %s' % train_std_post.shape[0])

#####################################################################################
# 6. PESQ-DNN Definition
#####################################################################################

INPUT_SHAPE = (None, input_length, 16, 2)

input_img = Input(shape=(INPUT_SHAPE))

c1 = TimeDistributed(Conv2D(n1, (W_k1, W_k1), padding='same', activation='relu'))(input_img)
x1 = TimeDistributed(MaxPooling2D((2, 2)))(c1)

c2 = TimeDistributed(Conv2D(n1, (W_k2, W_k2), padding='same', activation='relu'))(x1)
x2 = TimeDistributed(MaxPooling2D((2, 1)))(c2)

m1 = TimeDistributed(Conv2D(n2, (h_k, w1), padding='valid', activation='relu'))(x2)
m1 = TimeDistributed(MaxPooling2D((1, w_k_1)))(m1)

m2 = TimeDistributed(Conv2D(n2, (h_k, w2), padding='valid', activation='relu'))(x2)
m2 = TimeDistributed(MaxPooling2D((1, w_k_2)))(m2)

m3 = TimeDistributed(Conv2D(n2, (h_k, w3), padding='valid', activation='relu'))(x2)
m3 = TimeDistributed(MaxPooling2D((1, w_k_3)))(m3)

m4 = TimeDistributed(Conv2D(n2, (h_k, w4), padding='valid', activation='relu'))(x2)
m4 = TimeDistributed(MaxPooling2D((1, 1)))(m4)

m = Concatenate()([m1, m2, m3, m4])
m = Lambda(k_squeez_dim)(m)
m = Lambda(k_squeez_dim)(m)

lstm1 = Bidirectional(LSTM(LSTM_node, return_sequences=True),merge_mode='concat')(m)

block_out = TimeDistributed(Dense(32, activation='relu'))(lstm1)

block_out = TimeDistributed(Dense(1))(block_out)

Mos_predicted_block = TimeDistributed(Lambda(output_layer))(block_out)

att1 = TimeDistributed(Dense(64))(Mos_predicted_block)
att2 = TimeDistributed(Dense(1))(att1)
att3 = Softmax(axis=1)(att2)


att_pooling= Dot(axes=1)([att3, Mos_predicted_block])
att_pooling = Lambda(k_squeez_dim)(att_pooling)

fc4 = Dense(1)(att_pooling)
Mos_predicted = Lambda(output_layer)(fc4)

model = Model(inputs=input_img, outputs=Mos_predicted)
model.summary(line_length=150)

model.load_weights(model_name)

#####################################################################################
# 7. Inference
#####################################################################################
file_list = [f.name for f in os.scandir(databank_root) if f.name.endswith(file_extension)] # generate a list of .wav file in the audio directory

number_files = len(file_list)
predicted_PESQ = np.zeros((number_files, 1))

for file_idex, file in enumerate(file_list):
    print('Processing file: %s of %s' % (file_idex + 1, len(file_list)))

    coded_speech = sf.read(databank_root + file)[0] # Read the first coded speech utterance
    
    # Perform FFT of the current input speech utterance.
    signal_length = coded_speech.size
    num_frames = math.floor((signal_length / framing_params['frame_shift']) -1) # Calculate No. of frames

    dset_input = np.zeros((num_frames, int(framing_params['K_fft'])), dtype='float32') # shape of (No. of frames, 512) 
     
    offset = 0
    for frame_idx in range(num_frames):

        # compute non-redundant fft bins for current frame
        s_fft = get_fft_frame(coded_speech, offset, window, framing_params)
       
        dset_input[frame_idx, :] = (np.hstack((np.real(s_fft)[0:int(frame_length/2+1)], np.imag(s_fft[1:int(frame_length/2)])))).astype('float32') # concatenate 257 real frequency bins with 255 imaginary frequency bins

        offset = offset + framing_params['frame_shift']

    # Normalize the input of PESQ-DNN with provided statistics, obtained from the training data.

    inputs = (dset_input -train_mean_post)/ train_std_post # shape of inputs: (number of frames, 512)

    
    # Group the current input spectrum into blocks: No. of Blocks = No. of squence. Each block contains 16 frames. Two adjacent blocks have one frame overlap.

    real_seq_no=int(math.floor((num_frames-16)/15))
    batch_input_seq = np.zeros((1, int(real_seq_no), int(260), 16, 2), dtype='float32') # Input of PESQ-DNN, with shape of (Batch size, No. Sequence, 260, 16, 2)

    current_seq = np.transpose(inputs) # shape of (512, No. of frames)

    for seq_index in range(real_seq_no):            
        batch_input_seq[0, seq_index, 0:int(260 - 4 + 1), :, 0] = current_seq[:int(260 - 4 + 1), seq_index*15: seq_index*15+16]  # read 257 real frequency bins from current_seq
        batch_input_seq[0, seq_index, 1:int(260 - 4), :, 1] = current_seq[int(260 - 4 + 1):, seq_index*15: seq_index*15+16] # read 255 imaginary frequency bins from current_seq
        
        
    temp_output = model.predict(batch_input_seq) 
    predicted_PESQ[file_idex, 0] = temp_output[0, :]  # This is the output of the predicted PESQ socres for all files.
    
print('Mean PESQ Score: %s' % np.mean(predicted_PESQ))