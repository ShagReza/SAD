

clc
close all
clear all

folder = 'K:\SAD98\SadData2\Train\Temp\';


wavs = dir([folder,'*.wav']);

for i = 1: length(wavs)
    
    name = [folder, wavs(i).name];
    [wav, fs] = audioread(name);
    MatFeat= VAD_Drugman_ForPython(wav);
    
end

'ok';