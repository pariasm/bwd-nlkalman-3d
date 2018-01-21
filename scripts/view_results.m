load table

for s = [20],
	
	t = table;
	t = t(find(t(:,1) == s & t(:,3) == 1),:);
	t = t(:,5:end);

	ipsnr = 4;
	
	% find best parameters
	[psnr_max,imax] = max(t(:,ipsnr));
	best_prms = t(find(t(:,ipsnr) > psnr_max - 0.05*s),:);
	best_prms_ave = mean(best_prms);
	best_prms_std =  std(best_prms);
	
	disp('best parameters (wt, np, bx, psnr-deno)')
	best = sortrows(best_prms,-ipsnr);
	disp(best)
	
	disp('average and std. dev')
	disp(best_prms_ave)
	disp(best_prms_std)
end

%idx = find(t(:,2) <= 1.2 & t(:,2) >= 0.9 & t(:,4) >= 1.0 & t(:,4) < 1.5);
%t = t(idx,:);
%t = sortrows(t,-ipsnr);
%t1 = t;
%t1(idx,:) = [];
%plot3(t1(:,1),t1(:,3),t1(:,5),'r.'), hold on
%plot3(t(:,1),t(:,3),t(:,5),'b.'), hold off
figure(1)
scatter3(t(:,2),t(:,3),t(:,4),40,t(:,1),'filled'), hold on
hidden
hold off

xlabel('n'), ylabel('b'), box on
zlim([psnr_max-2.0 psnr_max+.5])
axis vis3d
view(90,0) % psnr vs bt

% optimal parameters
%
% FLAT PATCH 8x8x1 - LONGEST SEARCH REGION 7x7x4
%    dth beta
% 10 15.8 1.5 13.1 1.7
% 20 18.7 1.9 20.2 2.2 20.5 2
% 40 33   2.5

% FLAT PATCH 6x6x2 - LONGEST SEARCH REGION 7x7x4
%    dth beta
% 10 20.2 0.7
% 20 23.5 0.8
% 40 35.0 1.0


