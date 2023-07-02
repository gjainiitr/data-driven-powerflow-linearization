%{
case_name: string
Vm: 500x30
Va: 500x30
Ir: 500x30
Iim: 500x30
%}

function [Ir, Iim] = findI(case_name, Vm, Va)
mpc = loadcase(case_name);
start = mpc.branch(:,1);
last = mpc.branch(:,2);
r = mpc.branch(:,3);
x = mpc.branch(:,4);
num_bus = size(mpc.bus,1);
num_branch = size(mpc.branch,1);
Y = findY(start,last,r,x,num_bus);

%----------------------------%
%{
Vm = [500x30]
Va = [500x30]

%}

num_train = size(Vm,1);
Ir = zeros(num_train, num_bus);
Iim = zeros(num_train, num_bus);
for i=1:num_train


%%%%%------Need to edit---------%%%%%% 

Vr = zeros(num_bus,1);
Vim = zeros(num_bus,1);

% logic to convert into polar form
for j=1:num_bus
   [Vr(j),Vim(j)] =pol2cart(Va(i,j),Vm(i,j));
end

V = [Vr + Vim*i];
I = Y*V;

for k= 1:num_bus
Ir(i,k)  = real(I(k));
Iim(i,k) = imag(I(k));
end

%%%%---------------%

end
end