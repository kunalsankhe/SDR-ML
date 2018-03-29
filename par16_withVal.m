
clear all;
clc;

dist='14';
run='1';
for train_test=1:3 %(train 1 , test 2)

if (train_test==1) 
    M=000e3;
    N=200e3; 
 %    M=000e3;
  %   N=2000e3;
  %   N=2000e3;
elseif(train_test==2)
    M=200e3;
    N=10e3;
elseif(train_test==3)
    M=210e3;
    N=50e3;
   % M=2000e3;
   % N=50e3;
end

tic 
% case 1 7B and 7D
% case 2 65 and 70
% case 3 D89 and EFE
% case 4 D78 and D79
%device_name= {'X310_3123D58', 'X310_3123D64','X310_3123D70','X310_3123D78','X310_3123D76','X310_3123D79','X310_3123D80','X310_3123D7B', 'X310_3123D7E', 'X310_3123D54','X310_3124E4A'};
%device_name= {'X310_3123D58', 'X310_3123D76','X310_3124E4A','X310_3123D7E'}% Best 4 'X310_3123D78','X310_3123D76','X310_3123D79','X310_3123D80','X310_3123D7B', 'X310_3123D7E', 'X310_3123D54','X310_3124E4A'};
%device_name= {'X310_3123D58', 'X310_3123D76','X310_3124E4A','X310_3123D7E', 'X310_3123D64', 'X310_3123D54', 'X310_3123D70', 'X310_3123D78' } % Best 5
%, 'X310_3123D58', 'X310_3123D64','X310_3123D65','X310_3123D70','X310_3123D78','X310_3123D76','X310_3123D79','X310_3123D80','X310_3123D89'};
device_name={'X310_3123D7B', 'X310_3123D7D', 'X310_3123D7E', 'X310_3123D52', 'X310_3123D54', 'X310_3123D58', 'X310_3123D64', 'X310_3123D65', 'X310_3123D70', 'X310_3123D76', 'X310_3123D78', 'X310_3123D79', 'X310_3123D80', 'X310_3123D89', 'X310_3123EFE', 'X310_3124E4A'}
L = N;
w = 128;
K=0;
shft=1;
kk=(L-w+1)*w;
tmprxdispSym=zeros(N,1);
rxdisplaySym=zeros(kk,1);
MatrixData=zeros(kk*length(device_name),2);

for dev=1:length(device_name)
   file{dev}= strcat('WiFi_air_',device_name{dev},'_', dist, 'ft_run', run);
   x=load(file{dev}, 'wifi_rx_data');
   tmprxdispSym=x.wifi_rx_data(M+1:M+N);
   saen1=[];
    parfor k1 = 1:shft:L-w+1
        datawin(k1,:) = k1:k1+w-1;
        saen1(k1,:) = tmprxdispSym(datawin(k1,:));    
    end 
    rxdisplaySym=reshape(saen1.', [], 1);
    MatrixData((dev-1)*kk+1:dev*kk,:)=[real(rxdisplaySym) imag(rxdisplaySym)];
end

toc

if(train_test==1)
    savefile=strcat('IQ',num2str(length(device_name)),'_', dist, 'ft_train_',num2str(0.001*N*length(device_name)),'K.mat');
elseif(train_test==2) 
    savefile=strcat('IQ',num2str(length(device_name)),'_', dist, 'ft_validation_',num2str(0.001*N*length(device_name)),'K.mat');
elseif(train_test==3) 
    savefile=strcat('IQ',num2str(length(device_name)),'_', dist, 'ft_test_',num2str(0.001*N*length(device_name)),'K.mat');
end
save(savefile, 'MatrixData', '-v7.3' );

end
