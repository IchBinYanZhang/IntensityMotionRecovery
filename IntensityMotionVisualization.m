function IntensityMotionVisualization(f,u,v,e)

n_frames = size(f,3);
i = 1;
while 1
    if i==n_frames
        i=1;
    end;
    figure(1); imagesc(e(:,:,i));colorbar;drawnow;
    figure(2); imagesc(f(:,:,i));colorbar;caxis([0,1]);drawnow;
    flow = opticalFlow(u(:,:,i),v(:,:,i));
    figure(3);plot(flow,'ScaleFactor',200);set(gca,'ydir','reverse');drawnow;

    i = i+1;
end