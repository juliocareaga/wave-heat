%%%

close all; clear

% Colors:
clrB  = [ 170, 210, 250 ;
          252, 175, 173 ;
          250, 221, 162]./255;
clres = [ 0 0.4470 0.7410;
          0.8500 0.3250 0.0980;
          0.9290 0.6940 0.1250];

%-----------------------------------
modeltype = "W"; % chose "W" or "K"
%-----------------------------------
num   = 1
timed = {'4','8','16','32','64','128','256'};

NN   = [4,8,16,32,64,128,256,512];
ttau = 0.8./NN;

if modeltype == "K"
   %charfiles = {"errors_deg1_Euler_K", "errors_deg2_BDF2_K"};   
   charfiles = {"errors_deg1_Euler_K", "errors_deg2_BDF2_K", "errors_deg3_BDF2_K"};   
   h_degree  = [1,2,3];
   slopecte  = [0.6, 0.6, 0.6];      
               
   modeltitle = {"Kuznetsov"};
elseif modeltype == "W"
   %charfiles = {"errors_deg1_Euler_W", "errors_deg2_BDF2_W"};
   charfiles  = {"errors_deg1_Euler_W", "errors_deg2_BDF2_W", "errors_deg3_BDF2_W"};
   h_degree   = [1,2,3];
   slopecte   = [0.7,1.5,0.7]; % BDF2
   modeltitle = {"Westervelt"};
end 

for kk = 1:length(charfiles)

    eval(strcat("load('",charfiles{kk},"');"));
    %% Pressure:
    zz = L2error_ddp + L2error_dp + H1error_dp + L2error_pp + H1error_pp 
    %% Temperature
    zz = zz + L2error_dtheta + H1error_dtheta + L2error_theta + H1error_theta
    
    i = h_degree(kk)
    IMG1 = figure(99);
    [numspace, numtime] = size(zz);
    
    legaux = '{';
    j = numtime
    zzv = zz(:,j)
    
    loglog(hh, zzv, '-*','LineWidth',1,'Color',clres(kk,:)); hold on;
    grid on
    title(modeltitle,'interpreter','latex');
    set(gca,'Fontsize',17)
    lgd.NumColumns = 2;  
    
    for ll = 1:length(zz(:,numtime))-1
    
        xx1 = hh(ll);
        xx2 = hh(ll+1);
        yy1 = zz(ll,numtime);
        yy2 = zz(ll+1,numtime);
        
        dx = xx2-xx1;
        dy = yy2-yy1;
        xxm = [.5*(xx2+xx1), .5*(yy2+yy1)];
        slope = log(yy2/yy1)/log(xx2/xx1)
        th = text(xxm(1),xxm(2), strcat("$\bf ",num2str(slope),"$"),...
                  'Interpreter','latex','HorizontalAlignment', 'center','VerticalAlignment','middle',...
                  'BackgroundColor',clrB(kk,:), 'String','boldString', 'FontSize', 8)
    end
    
end
%legend({'Euler, $\eta=1$','BDF2, $\eta=2$'},'location','southeast', 'Interpreter','latex', 'Fontsize',14);
legend({'Euler, $\eta=1$','BDF2, $\eta=2$','BDF2, $\eta=3$'},'location','southeast', 'Interpreter','latex', 'Fontsize',14);

ylabel('$\mathbf{E}_{\tau}(T)$','interpreter','latex','Fontsize',24);
xlabel('$h$','interpreter','latex','Fontsize',24);    
saveas(gcf,strcat('errorplot_',modeltype,'.png'))
%------------------------------------------------------------------
