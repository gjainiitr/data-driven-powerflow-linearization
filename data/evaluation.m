mpc = load('data/_data.mat');
data = mpc.data;

% Forward normal error
[Xp, Xq] = RegressionForward(1, bus, data);
[Perror, Qerror] = ForwardError(Xp, Xq, num_train, data)
Perror
Qerror

% Inverse normal error
[Xva, Xv] = RegressionInverse(1, num_load, data, ref)
[Verror, Vaerror] = InverseError(Xv, Xva, num_train, data)
Verror
Vaerror

% Forward TTS
[Xp, Xq, error] = ForwardErrorTTS(bus, eg, data, 0.2, 1)
error.Perror
error.Qerror

% Inverse TTS
[Xv, Xva, error] = InverseErrorTTS(num_load, num_train, data, 0.2, 1, ref)
error.Verror
error.Vaerror
