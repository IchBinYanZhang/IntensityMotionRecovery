%%% event-visualization
function EventVisualization(x,y,pol,time,triggers)

% normalize the time stamps
time = time-time(1);
pol = double(pol)*256;
x = x+1;
y = y+1;
% partation the temporal domain into N cells
ht = 10^4;  % temporal grid size; how many events are shown per frame
N = round(length(time)/ht);  % the display will show 300 frames


% main loop for frame display
frame = 128*ones(128,128);
for i = 1:N
    idx_pre = ht*(i-1)+1;
    idx_cur = ht*i;
    xx = x(idx_pre:idx_cur);
    yy = y(idx_pre:idx_cur);
    pp = pol(idx_pre:idx_cur);
    fprintf('i = %d\n',i);
    for ii = 1:length(xx)
        frame(xx(ii),yy(ii)) = pp(ii);
    end
    
    %figure(1);imshow(frame);drawnow;pause(0.001);
end

      
    
    

