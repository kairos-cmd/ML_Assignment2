clear all, close all,

% True position inside unit circle
radius = rand; theta = 2*pi*rand;
pTrue = [radius*cos(theta);radius*sin(theta)];  

for K = 1:40 % for each specified number of landmarks
    % Landmark positions evenly spaces on the unit circle 
    radius = 1; theta = [0,2*pi/K*[1:(K-1)]];
    pLandmarks = [radius*cos(theta);radius*sin(theta)];

    % Generate range measurements
    deviation = 3e-2*ones(1,K);%noise standard deviation=0.3
    r = sqrt(sum((repmat(pTrue,1,K)-pLandmarks).^2,1)) + deviation.*randn(1,K);

    % Parameters of the prior 
    sigmax = 2.5e-1;
    sigmay = sigmax;

    % Evaluate the MAP estimation objective function on a grid
    Nx = 100; Ny = 101;
    %range of horizontal and vertical coordinates
    xGrid = linspace(-2,2,Nx); 
    yGrid = linspace(-2,2,Ny);
    [h,v] = meshgrid(xGrid,yGrid);
    MAPobjective = (h(:)/sigmax).^2 + (v(:)/sigmay).^2;

    for i = 1:K
        di = sqrt((h(:)-pLandmarks(1,i)).^2+(v(:)-pLandmarks(2,i)).^2);
        MAPobjective = MAPobjective + ((r(i)-di)/deviation(i)).^2;
    end
    zGrid = reshape(MAPobjective,Ny,Nx);
    figure(K);
    plot(pLandmarks(1,:),pLandmarks(2,:),'o'); axis([-2 2 -2 2]),
    % Display the MAP objective contours
    minV = min(MAPobjective); maxV = max(MAPobjective);
    values = minV + (sqrt(maxV-minV)*linspace(0.1,0.9,21)).^2;
    contour(xGrid,yGrid,zGrid,values); xlabel('x'), ylabel('y'),
    legend('landmark location of the object',' true location of the object','MAP objective function contours'), 
    title(strcat({'MAP Objective for K = '},num2str(K)));
    grid on, axis equal,
end
