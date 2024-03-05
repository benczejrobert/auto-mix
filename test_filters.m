clear all
clc
close all

% Fs  = 44.1e3;
[x,Fs] = audioread('D:\PCON\Disertatie\AutoMixMaster\datasets\diverse-test\white-noise-mono.wav');
N   = 2;
G   = 24; % +24 la 12k si -40 la 100 Hz
Q   = 2;
f_c = 12000; % Notch at f_c Hz
Wo  = f_c/(Fs/2); 
BW  = Wo/Q; % Bandwidth will occur at -3 dB for this special case
[B1,A1] = designParamEQ(N,G,Wo,BW,'Orientation','row'); % -inf thing
[NUM,DEN]  = iirnotch(Wo,BW); % or [NUM,DEN] = designParamEQ(2,G,Wo,BW); % - does not work with this
BQ1 = dsp.SOSFilter('Numerator',B1,'Denominator',A1);
BQ2 = dsp.SOSFilter('Numerator',NUM,'Denominator',DEN);
hfvt = fvtool(BQ1,BQ2,'Fs',Fs,'FrequencyScale','Log','Color','white');
legend(hfvt,'8th order notch filter','2nd order notch filter');

y=BQ1(x); % - filter loaded signal then write it
audiowrite('eq-ed_12k_matlab.wav',y,Fs);
% TODO howto apply filter to a signal

%%

clear all
clc
close all

% Fs  = 44.1e3;
[x,Fs] = audioread('D:\PCON\Disertatie\AutoMixMaster\datasets\diverse-test\white-noise-mono.wav');
N   = 2;
G   = -40; % +24 la 12k si -40 la 100 Hz
Q   = 2;
f_c = 100; % Notch at f_c Hz
Wo  = f_c/(Fs/2); 
BW  = Wo/Q; % Bandwidth will occur at -3 dB for this special case
[B1,A1] = designParamEQ(N,G,Wo,BW,'Orientation','row'); % -inf thing
[NUM,DEN]  = iirnotch(Wo,BW); % or [NUM,DEN] = designParamEQ(2,G,Wo,BW); % - does not work with this
BQ1 = dsp.SOSFilter('Numerator',B1,'Denominator',A1);
BQ2 = dsp.SOSFilter('Numerator',NUM,'Denominator',DEN);
hfvt = fvtool(BQ1,BQ2,'Fs',Fs,'FrequencyScale','Log','Color','white');
legend(hfvt,'8th order notch filter','2nd order notch filter');

y=BQ1(x); % - filter loaded signal then write it
audiowrite('eq-ed_100_matlab.wav',y,Fs);
% TODO howto apply filter to a signal

%%
% TODO howto cascade
clear all
clc
close all

% Fs  = 44.1e3;
[x,Fs] = audioread('D:\PCON\Disertatie\AutoMixMaster\datasets\diverse-test\white-noise-mono.wav');
N   = [2,2]; % try 2 - order
G   = [-40,12]; % +24 la 12k si -40 la 100 Hz
Q   = 2;
f_c = [100,12000]; % Notch at f_c Hz
Wo1  = 100/(Fs/2);
Wo2  = 12000/(Fs/2);
Wo = [Wo1,Wo2];
BW  = [Wo1/Q,Wo2/Q]; % Bandwidth will occur at -3 dB for this special case
[B1,A1] = designParamEQ(N,G,Wo,BW,'Orientation','row'); % -inf thing
[NUM,DEN]  = iirnotch(Wo1,BW(1)); % or [NUM,DEN] = designParamEQ(2,G,Wo,BW); % - does not work with this
BQ1 = dsp.SOSFilter('Numerator',B1,'Denominator',A1);
BQ2 = dsp.SOSFilter('Numerator',NUM,'Denominator',DEN);
hfvt = fvtool(BQ1,BQ2,'Fs',Fs,'FrequencyScale','Log','Color','white');
legend(hfvt,'8th order notch filter','2nd order notch filter');

y=BQ1(x); % - filter loaded signal then write it
audiowrite('eq-ed_100-12k_matlab.wav',y,Fs);
