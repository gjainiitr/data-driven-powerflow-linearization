mpc = load('30data.mat');
data = mpc.data;

writematrix(data.P,'30P.csv') ;
writematrix(data.Q,'30Q.csv') ;
writematrix(data.Va,'30Va.csv') ;
writematrix(data.V,'30V.csv') ;

mpc = load('39data.mat');
data = mpc.data;

writematrix(data.P,'39P.csv') ;
writematrix(data.Q,'39Q.csv') ;
writematrix(data.Va,'39Va.csv') ;
writematrix(data.V,'39V.csv') ;

mpc = load('57data.mat');
data = mpc.data;

writematrix(data.P,'57P.csv') ;
writematrix(data.Q,'57Q.csv') ;
writematrix(data.Va,'57Va.csv') ;
writematrix(data.V,'57V.csv') ;

mpc = load('118data.mat');
data = mpc.data;

writematrix(data.P,'118P.csv') ;
writematrix(data.Q,'118Q.csv') ;
writematrix(data.Va,'118Va.csv') ;
writematrix(data.V,'118V.csv') ;

mpc = load('300data.mat');
data = mpc.data;

writematrix(data.P,'300P.csv') ;
writematrix(data.Q,'300Q.csv') ;
writematrix(data.Va,'300Va.csv') ;
writematrix(data.V,'300V.csv') ;