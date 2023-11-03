# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:27:59 2019

@author: Administrator
"""

####====================================================
import numpy as np
import librosa
from glob import glob 
from scipy import signal
import h5py
from scipy import interpolate
import os

import config as c
import utils as u

from librosa import util
#import python_speech_features as psf

#import matlab.engine
#eng = matlab.engine.start_matlab() 

####====================================================
def numframes_context(feat_path):
#   feat_list = glob(feat_path + '\*.npy')
   feat_list = list(feat_path.glob('**/*.npy')) 
   
   num_frames = 0
   for i, ft in enumerate(feat_list):
       X = np.load(ft)
       frames = range(c.context+c.offset, X.shape[1]-c.context-1, c.hop_frames)  
       num_frames += len(frames)
       print('#frames until file {} : {}'.format(i,num_frames))
   return num_frames
    
####====================================================
def h5file(file_name, data_shape, NumSamples):
    
#    train_shape = (NumSamples, flag.FFTSIZE//2, flag.SegSIZE, 1)    
    output_file = h5py.File(file_name, mode='a')
    output_file.create_dataset("X", data_shape, np.float32, maxshape=(None, data_shape[1], data_shape[2], 1))
    
#    nrz_shape = (NumSamples, 2, flag.FFTSIZE//2+1)
#    output_file.create_dataset("Nrz", nrz_shape, np.float32, maxshape=(None, 2, flag.FFTSIZE//2+1))
    
    return output_file  

####==================================================== 
def resample_wav(wav, orig_sr, target_sr=8000):
    if orig_sr != target_sr:
        wav = librosa.resample(wav, orig_sr, target_sr)
    return wav
        

####====================================================
def normalize_wav(wav, nrz_max=False, nrz_mean=False):
    """written by hrbk 98/08/13
    """
    if c.nrz_max:
        wav /= max(abs(wav))
        
    if c.nrz_mean:
        wav -= np.mean(wav)
    
    return wav

####====================================================
def preemphasis(wav, coeff=0.975):
    """
    Emphasize high frequency range of the waveform by increasing power(squared amplitude).

    Parameters
    ----------
    wav : np.ndarray [shape=(n,)]
        Real-valued the waveform.

    coeff: float <= 1 [scalar]
        Coefficient of pre-emphasis.

    Returns
    -------
    preem_wav : np.ndarray [shape=(n,)]
        The pre-emphasized waveform.
    """
    preem_wav = signal.lfilter([1, -coeff], [1], wav)
    return preem_wav

####====================================================

def equalize_wav(path, target_sr, eq_win, eq_shift):
    
    wav, orig_sr = librosa.load(path, target_sr)
    
    frms = psf.sigproc.framesig(wav, eq_win*target_sr, eq_shift*target_sr)
    r = frms.std(axis=-1, keepdims=True)
    frms = frms/r
    sig1 = psf.sigproc.deframesig(frms, siglen=len(wav), 
                              frame_len=eq_win*target_sr, frame_step= eq_shift*target_sr)

    wav = sig1 / max(abs(sig1))
    
    return wav
    
####====================================================
def preprocess_wav(path, target_sr, nrz_max, nrz_mean):
    """written by hrbk 98/08/13
    """
    if c.apply_eq == 1:
        wav = equalize_wav(path, target_sr, c.eq_win, c.eq_shift)
    else:    
        wav, orig_sr = librosa.load(path, target_sr) ## Audio will be automatically resampled to the given rate
    
    wav = resample_wav(wav, orig_sr, target_sr) ## for more assurance, hrbk added this line, becuse the previous line probably resample the input signal
    
    wav = normalize_wav(wav, nrz_max=False, nrz_mean=False)
    
    wav_preproc = preemphasis(wav, coeff=0.975)
    
#    wav_preproc = librosa.effects.preemphasis(wav, coef=0.97)
        
    return wav_preproc

####====================================================
def VAD_librosa(path):
    """written by hrbk 98/08/13
    """
    wav, _ = librosa.load(path, c.sample_rate) ## Audio will be automatically resampled to the given rate
    
    if c.nrz_mean:
        wav -= np.mean(wav)
        
    vad = librosa.effects.split(wav, top_db=c.top_db_vad, frame_length=int(c.win_length), 
                        hop_length=int(c.hop_length))
    
    isig = []
    for v in vad:
        isig.append(np.arange(*v))
    isig = np.concatenate(isig, axis=0)
    isil = np.setdiff1d(np.arange(len(wav)), isig)    
    
    return wav, isig, isil
####==================================================== 
def MVN(X):
    """written by hrbk 98/08/13
        X has the shape of (number of features, timeframes), exp: (129 frqbin, 1500 frames)
    """
    mu = X.mean(axis=1).reshape(-1,1)  ## mu is (129,1)
    sd = X.std(axis=1).reshape(-1,1)+1e-12  ## sd is (129,1)
    Xn = (X-mu)/sd
    return Xn, mu, sd

####====================================================
def MVN_general(X, ax):
    """written by hrbk 98/08/13
        X has the shape of (number of features, timeframes), exp: (129 frqbin, 1500 frames)
    """
    mu = X.mean(axis=ax)
    sd = X.std(axis=ax)+1e-12  ## sd is (129,1)
    Xn = (X-mu)/sd
    return Xn, mu, sd

####====================================================
def wave2spec(path):
    """written by hrbk 98/08/13
    """
    wav        = preprocess_wav(path, c.sample_rate, c.nrz_max, c.nrz_mean)    
    stft       = librosa.stft(y=wav, n_fft=c.n_fft, hop_length=c.hop_length, win_length=c.win_length, center=c.center_lib)
    mag_spec, phase = librosa.magphase(stft, power=1)     
    mag_spec = np.where(mag_spec>c.min_mag, mag_spec, c.min_mag)    
    
    if c.amp == 'log':
        if c.task == 'class':
            mag_spec = librosa.amplitude_to_db(mag_spec)  ## equivalent to 20*log10*(mag_spec) or 10*log10(pow_spec)
        elif c.task == 'enhan':
            mag_spec = np.log10(mag_spec) 
        
    mag_spec = np.float32(mag_spec)
    
    ## write to .npy file
#    full_wav_name = path.replace('\\','/')
#    raw_name = full_wav_name.split('/')[-1][:-4]+'.npy'    
#    out_path = os.path.join(c.feat_folder , raw_name)
#    np.save(out_path, mag_spec)
    
#    np.save(path.replace('.wav', '.npy'), mag_spec)   
    
    return mag_spec, phase
    
    
####====================================================
def wave2melspec(path):
    """written by hrbk 98/08/13
    """
    wav      = preprocess_wav(path, c.sample_rate, c.nrz_max, c.nrz_mean)    
    stft     = librosa.stft(y=wav, n_fft=c.n_fft, hop_length=c.hop_length, win_length=c.win_length, window=c.window, center=c.center_lib)
    mag_spec, _   = librosa.magphase(stft, power=1)
    mag_spec = np.where(mag_spec>c.min_mag, mag_spec, c.min_mag)   
    
    mag_spec_mel = np.dot(c.mel_basis, mag_spec)    
    mag_spec_mel = np.where(mag_spec_mel>c.min_mag, mag_spec_mel, c.min_mag) 
    
    if c.amp == 'log':
        if c.task == 'class':
            spec_mel_db = librosa.amplitude_to_db(mag_spec_mel)
        elif c.task == 'enhan':
            spec_mel_db = np.log10(mag_spec_mel) 
            
    elif c.amp == 'pow':
        spec_mel_db = mag_spec_mel**(2/15)
    
    else:
        spec_mel_db = mag_spec_mel
     
    spec_mel_db = np.float32(spec_mel_db)
#    return spec_mel_db
    np.save(path.replace('.wav','.npy'), spec_mel_db)


####====================================================
def wave2mfcc(path):
    """written by hrbk 98/08/13
    """
    wav         = preprocess_wav(path, c.sample_rate, c.nrz_max, c.nrz_mean)    
    stft        = librosa.stft(y=wav, n_fft=c.n_fft, hop_length=c.hop_length, win_length=c.win_length, window=c.window, center=c.center_lib)
    mag_spec, _ = librosa.magphase(stft, power=1)   
    mag_spec = np.where(mag_spec>c.min_mag, mag_spec, c.min_mag)
    
    mag_spec_mel = np.dot(c.mel_basis, mag_spec)    
    spec_mel_db  = librosa.amplitude_to_db(mag_spec_mel)
    mfcc         = librosa.feature.mfcc(S=spec_mel_db, n_mfcc=c.n_mfcc) ## equal to mfcc = np.dot(librosa.filters.dct(c.n_mfcc, spec_mel_db.shape[0]), spec_mel_db)
#    mfcc = np.dot(librosa.filters.dct(c.n_mfcc, spec_mel_db.shape[0]), spec_mel_db)
    mfcc = np.float32(mfcc)
    return mfcc


##==================================================================================    
def wav2fullmelspec(path):   
    
    wav        = preprocess_wav(path, c.sample_rate, c.nrz_max, c.nrz_mean)    
    stft       = librosa.stft(y=wav, n_fft=c.n_fft, hop_length=c.hop_length, win_length=c.win_length, window=c.window, center=c.center_lib)
    mag_spec, phase = librosa.magphase(stft, power=1)     
    mag_spec = np.where(mag_spec>c.min_mag, mag_spec, c.min_mag)
    
#    mag_spec_mel = np.empty((mag_spec.shape[0], mag_spec.shape[1]), dtype=np.float32)  
    mag_spec_mel = []
    for i in range(0,mag_spec.shape[1]):        
        x0 = mag_spec[:,i]       
        tck = interpolate.splrep(u.hz_points, x0)        
        z1 = interpolate.splev(u.mel_points_hz, tck)        
        z1 = np.where(z1>c.min_mag, z1, c.min_mag)        
#        mag_spec_mel[:,i] = z1
        mag_spec_mel.append(np.expand_dims(z1,1))
    
    mag_spec_mel = np.concatenate(mag_spec_mel, axis=1)
    if c.amp == 'log':
        if c.task == 'class':
            spec_mel_db = librosa.amplitude_to_db(mag_spec_mel)  ## equivalent to 20*log10*(mag_spec) or 10*log10(pow_spec)
        elif c.task == 'enhan':
            spec_mel_db = np.log10(mag_spec_mel) 

    elif c.amp == 'pow':
        spec_mel_db = mag_spec_mel**(2/15)
    else:
        spec_mel_db = mag_spec_mel
     
    spec_mel_db = np.float32(spec_mel_db)
        
    
    np.save(path.replace('.wav', '.npy'), spec_mel_db) 
    
#    return spec_mel_db

##==================================================================================
def provide_context_feats(feat_path):
    
#   feat_list = glob(feat_path + '\*.npy')
   feat_list = list(feat_path.glob('**/*.npy')) 
   
   Xn_all = []   
       
   for i, ft  in enumerate(feat_list):
        ## load features of speech utterance for example: 129fbin*5000frames and apply mvn
       X = np.load(ft)  
#       print(' Extracting/Appending Context Features for Name {} '.format(ft))
#       print(' Extracting/Appending Context Features for File {}/{} '.format(i+1, len(feat_list)))
#       print(' Shape is {} '.format(X.shape[1]))
       
           
       if c.mvn_type == 'utt':           
           Xn,_,_ = MVN(X)
       elif c.mvn_type == 'no':
               Xn = X
               
       elif c.mvn_type == 'win':
           nfrm = X.shape[1]
           
           if nfrm <= c.cnx_mvn:
              Xn,_,_ = MVN(X) 
           
           else:
               Xn = []
               for j0 in range(nfrm):
                   if j0<c.cnx_mvn//2:
                       X_mvn = X[0:c.n_fft//2, 0 : c.cnx_mvn+1]
                       X_mvn,_,_ = MVN(X_mvn)
                       Xn.append(X_mvn[:,j0].reshape(-1,1))
                       
                   elif (j0>=c.cnx_mvn//2) and (j0<nfrm-(c.cnx_mvn//2)):
                       X_mvn = X[0:c.n_fft//2, j0-(c.cnx_mvn//2) : j0+(c.cnx_mvn//2)+1]
                       X_mvn,_,_ = MVN(X_mvn)
                       Xn.append(X_mvn[:,c.cnx_mvn//2].reshape(-1,1))
                       
                   else:
                       X_mvn = X[0:c.n_fft//2, nfrm-c.cnx_mvn : nfrm+1]
                       X_mvn,_,_ = MVN(X_mvn)
                       Xn.append(X_mvn[:,j0-nfrm].reshape(-1,1))
                   
               Xn = np.concatenate(Xn, axis=1)
           
       ## select number of frames with the intervals of hop_frames      
       frames = range(c.context+c.offset, Xn.shape[1]-c.context-1, c.hop_frames)
       Xn_utt = []
       for j in range(len(frames)):
           if c.contex_type == 'odd':
               Xn_utt.append(Xn[0:c.n_fft//2, frames[j]-c.context : frames[j]+c.context+1])
           else:
               Xn_utt.append(Xn[0:c.n_fft//2, frames[j]-c.context : frames[j]+c.context])
       del Xn
       
       if len(Xn_utt)>0:
#           Xn = np.asarray(Xn_utt).reshape(len(Xn_utt), c.n_fft//2, -1, 1)
           Xn = np.asarray(Xn_utt).reshape(len(Xn_utt), Xn_utt[0].shape[0], Xn_utt[0].shape[1], 1)
           del Xn_utt
           Xn = np.float32(Xn) ## recently added np.float32 1/19/2020
           
           ## append context feature of utterances    
           Xn_all.append(Xn) #   Xn_all = np.vstack((Xn_all, Xn))
           
           del Xn
           
#   Xn_all = np.float32(Xn_all)
    
   Xn_all = np.concatenate(Xn_all, axis=0)
   return Xn_all       

##==================================================================================       
def provide_context_feats_par(feat_path):    

        ## load features of speech utterance for example: 129fbin*5000frames and apply mvn
   X = np.load(feat_path)  
#       print(' Extracting/Appending Context Features for Name {} '.format(ft))
#       print(' Extracting/Appending Context Features for File {}/{} '.format(i+1, len(feat_list)))
#       print(' Shape is {} '.format(X.shape[1]))
   
       
   if c.mvn_type == 'utt':           
       Xn,_,_ = MVN(X)
   elif c.mvn_type == 'no':
           Xn = X
           
   elif c.mvn_type == 'win':
       nfrm = X.shape[1]
       
       if nfrm <= c.cnx_mvn:
          Xn,_,_ = MVN(X) 
       
       else:
           Xn = []
           for j0 in range(nfrm):
               if j0<c.cnx_mvn//2:
                   X_mvn = X[0:c.n_fft//2, 0 : c.cnx_mvn+1]
                   X_mvn,_,_ = MVN(X_mvn)
                   Xn.append(X_mvn[:,j0].reshape(-1,1))
                   
               elif (j0>=c.cnx_mvn//2) and (j0<nfrm-(c.cnx_mvn//2)):
                   X_mvn = X[0:c.n_fft//2, j0-(c.cnx_mvn//2) : j0+(c.cnx_mvn//2)+1]
                   X_mvn,_,_ = MVN(X_mvn)
                   Xn.append(X_mvn[:,c.cnx_mvn//2].reshape(-1,1))
                   
               else:
                   X_mvn = X[0:c.n_fft//2, nfrm-c.cnx_mvn : nfrm+1]
                   X_mvn,_,_ = MVN(X_mvn)
                   Xn.append(X_mvn[:,j0-nfrm].reshape(-1,1))
               
           Xn = np.concatenate(Xn, axis=1)
       
   ## select number of frames with the intervals of hop_frames      
   frames = range(c.context+c.offset, Xn.shape[1]-c.context-1, c.hop_frames)
   Xn_utt = []
   for j in range(len(frames)):
       if c.contex_type == 'odd':
           Xn_utt.append(Xn[0:c.n_fft//2, frames[j]-c.context : frames[j]+c.context+1])
       else:
           Xn_utt.append(Xn[0:c.n_fft//2, frames[j]-c.context : frames[j]+c.context])
   del Xn
   
   if len(Xn_utt)>0:
#           Xn = np.asarray(Xn_utt).reshape(len(Xn_utt), c.n_fft//2, -1, 1)
       Xn = np.asarray(Xn_utt).reshape(len(Xn_utt), Xn_utt[0].shape[0], Xn_utt[0].shape[1], 1)
       del Xn_utt
   else:
       Xn = np.empty((0,X.shape[0],c.context*2,1), dtype = 'float32')
    
   return np.float32(Xn)  ## recently added np.float32 1/19/2020

          
##==================================================================================
def wav2drugman(path):  
    wav        = preprocess_wav(path, c.sample_rate, c.nrz_max, c.nrz_mean) 
    WavToList = wav.tolist();
    WavToFloat=[float(x) for x in WavToList]
    MatFeat = eng.VAD_Drugman_ForPython(WavToFloat)
    MatFeatNp=np.asarray(MatFeat, dtype='float32')
    np.save(path.replace('.wav', '.npy'), MatFeatNp) 
##==================================================================================
# Compute a chromagram from a waveform or power spectrogram.
def wav2chroma(path):
    #path='G:\\a.wav'
    #wav        = SF.preprocess_wav(path, c.sample_rate, c.nrz_max, c.nrz_mean)
    wav        = preprocess_wav(path, c.sample_rate, c.nrz_max, c.nrz_mean)     
    ChromaFeatNp=librosa.feature.chroma_stft(y=wav, n_fft=c.n_fft, hop_length=c.hop_length, win_length=c.win_length, window=c.window, center=c.center_lib)
    #ChromaFeatNp_cqtt=librosa.feature.chroma_cqtt(y=wav,n_chroma=12, sr=c.sample_rate,hop_length=80,win_length=256)
    #ChromaFeatNp_cqtt=librosa.feature.chroma_cens(y=wav,n_chroma=12, sr=c.sample_rate,hop_length=80,win_length=256)
    
    if c.nrm_chroma != np.inf:
        if c.amp == 'log':
            ChromaFeatNp = np.log10(ChromaFeatNp) 
        elif c.amp == 'pow':
            ChromaFeatNp = ChromaFeatNp**(2/15) 
            
#            ChromaFeatNp = util.normalize(ChromaFeatNp, norm=np.inf, axis=0)           
            
    ChromaFeatNp = np.float32(ChromaFeatNp)  ## recently added this line 1/19/2020
    np.save(path.replace('.wav', '.npy'), ChromaFeatNp) 
##==================================================================================
def wav2sadfeats(path):
    
    wav        = preprocess_wav(path, c.sample_rate, c.nrz_max, c.nrz_mean)
    spec_cent = librosa.feature.spectral_centroid(y=wav, n_fft=c.n_fft, sr=c.sample_rate, hop_length=c.hop_length, win_length=c.win_length, center=c.center_lib, window=c.window)
    spec_band = librosa.feature.spectral_bandwidth(y=wav, n_fft=c.n_fft, sr=c.sample_rate, hop_length=c.hop_length, win_length=c.win_length, center=c.center_lib, window=c.window)
    spec_flat = librosa.feature.spectral_flatness(y=wav, n_fft=c.n_fft, hop_length=c.hop_length, win_length=c.win_length, center=c.center_lib, window=c.window)
    spec_roll = librosa.feature.spectral_rolloff(y=wav, n_fft=c.n_fft, sr=c.sample_rate, hop_length=c.hop_length, win_length=c.win_length, center=c.center_lib, window=c.window)
    zcr       = librosa.feature.zero_crossing_rate(y=wav, frame_length=c.win_length, hop_length=c.hop_length, center=c.center_lib)
    spec_cont =librosa.feature.spectral_contrast(y=wav, n_fft=c.n_fft, sr=c.sample_rate, hop_length=c.hop_length, win_length=c.win_length, center=c.center_lib, window=c.window,fmin=100.0, n_bands=6)
    
    z = np.concatenate((spec_cent,spec_band), axis=0)
    z = np.concatenate((z,spec_flat), axis=0)
    z = np.concatenate((z,spec_roll), axis=0)
    z = np.concatenate((z,zcr), axis=0)
    z = np.concatenate((z,spec_cont), axis=0)
    
    sadfeats = np.float32(z)
    np.save(path.replace('.wav', '.npy'), sadfeats) 
    


##==================================================================================
def provide_context_feats_test(ft):   
    
    ## load features of speech utterance for example: 129fbin*5000frames and apply mvn
   X = np.load(ft[0])
   
   if c.mvn_type == 'utt':           
       Xn,_,_ = MVN(X)
   elif c.mvn_type == 'no':
           Xn = X
           
   elif c.mvn_type == 'win':
       nfrm = X.shape[1]
       
       if nfrm <= c.cnx_mvn:
          Xn,_,_ = MVN(X) 
       
       else:
           Xn = []
           for j0 in range(nfrm):
               if j0<c.cnx_mvn//2:
                   X_mvn = X[0:c.n_fft//2, 0 : c.cnx_mvn+1]
                   X_mvn,_,_ = MVN(X_mvn)
                   Xn.append(X_mvn[:,j0].reshape(-1,1))
                   
               elif (j0>=c.cnx_mvn//2) and (j0<nfrm-(c.cnx_mvn//2)):
                   X_mvn = X[0:c.n_fft//2, j0-(c.cnx_mvn//2) : j0+(c.cnx_mvn//2)+1]
                   X_mvn,_,_ = MVN(X_mvn)
                   Xn.append(X_mvn[:,c.cnx_mvn//2].reshape(-1,1))
                   
               else:
                   X_mvn = X[0:c.n_fft//2, nfrm-c.cnx_mvn : nfrm+1]
                   X_mvn,_,_ = MVN(X_mvn)
                   Xn.append(X_mvn[:,j0-nfrm].reshape(-1,1))
               
           Xn = np.concatenate(Xn, axis=1)

   Xn_utt = []
   for i in range(Xn.shape[1]):
       
       if c.contex_type == 'even':
           
           if i<c.context_test:
               Xn_utt.append(Xn[:, 0 : 2*c.context_test])
               
           elif (i>=c.context_test) and (i<Xn.shape[1]-c.context_test):
               Xn_utt.append(Xn[:, i-c.context_test : i+c.context_test])
               
           else:
               Xn_utt.append(Xn[:, Xn.shape[1]-2*c.context_test : Xn.shape[1]])   
   del Xn
       
   if len(Xn_utt)>0:
#           Xn = np.asarray(Xn_utt).reshape(len(Xn_utt), c.n_fft//2, -1, 1)
       Xn = np.asarray(Xn_utt).reshape(len(Xn_utt), Xn_utt[0].shape[0], Xn_utt[0].shape[1], 1)
       del Xn_utt 
   else:
       Xn = np.empty((0,X.shape[0],c.context_test*2,1))

   Xn = np.float32(Xn)
   
#   cnx_path = os.path.join(Feat_Cnx_path, 'Cnx_'+ft.split('\\')[-1]) 
#   np.save(cnx_path, Xn)
#   print(ft[1])
   np.save(ft[1], Xn)
#   return Xn   
       
       
##==================================================================================

