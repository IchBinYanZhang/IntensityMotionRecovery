function [ x, y, polarity, time, triggers ] = convertAERtoMAT( filepath )
%CONVERTAERTOMAT loads a given tmpdiff128 .aedat, converts it and saves it as .mat
%  x, y are pixel positions in the range [0, 127]
%  polarity of '1' means that an ON event was detected, otherwise it's OFF
%  time is a timestamp in microseconds

% create output file path
extensionPosition = find(filepath == '.', 1, 'last');
if length(filepath) <= extensionPosition,
    error('jaerMATLAB:convertAERtoMAT', 'Filepath %s has no valid extension type!', filepath);
end
outputFilepath = [filepath(1:extensionPosition), 'mat'];

% load AER file
[address, time] = loadaerdat(filepath, 4*10^6);

% extract positions and polarity 
if strcmp(class(address), 'uint32')
  % same as for 16 bit addresses?
    % x is the 7 bits from position 8 to 1
    x = uint8( bitand(bitshift(address, -1), hex2dec('7f')) );
    % y is the 7 bits starting from position 9
    y = uint8( bitand(bitshift(address, -8), hex2dec('7f')) );
    % polarity is the last bit 
    polarity = uint8( 1 - bitand(address, 1) );
    % triggers are marked by bit 15
    triggers = (bitand(address, hex2dec('8000')) ~= 0);
elseif strcmp(class(address), 'uint16')
    % x is the 7 bits from position 8 to 1
    x = uint8( bitand(bitshift(address, -1), hex2dec('7f')) );
    % y is the 7 bits starting from position 9
    y = uint8( bitand(bitshift(address, -8), hex2dec('7f')) );
    % polarity is the last bit 
    polarity = uint8( 1 - bitand(address, 1) );
    % triggers are marked by bit 15
    triggers = (bitand(address, hex2dec('8000')) ~= 0);
else
    error('convertAERtoMAT() found invalid addresses. Only types uint16 and uint32 are supported.');
end


% save results in MAT file
res.names = {'X', 'Y', 'ON/OFF', 'TIMEus', 'TRG'};
res.values = {x, y, polarity, time, triggers};
save(outputFilepath, '-struct', 'res');

end

