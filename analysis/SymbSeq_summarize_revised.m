close all; clear all;

groups = {'A','B','C','D'};
group_labels = {'Response Group','Visual Group','Combined','Control'};
%groups = {'C'};
data_dir = '../data';
data_indx = 1;
sess_indx = 1;

REPROCESS = 0;

if REPROCESS

for c = 1:length(groups)

	group_ID = groups{c};

	%muRT = []; stdRT = []; acc = [];
	deltaRT = []; deltaAcc = []; deltaSTD = [];
	%acf = [];

	switch group_ID
	case {'A','B'};
		files = dir(fullfile(data_dir, sprintf('*%s-*SymbSeq*.mat',group_ID)));
	otherwise
		files = dir(fullfile(data_dir, sprintf('*_%s_*SymbSeq*.mat',group_ID)));
	end;

	for f = 1:length(files);
		file_str = files(f).name;

		switch group_ID
		case 'D'
			SN(f) = double(str2num(file_str(2:4)));
		otherwise
			SN(f) = double(str2num(file_str(1:3)));
		end;

		switch group_ID;
		case {'A','B'}
			date_str{f} = files(f).name(end-27:end-17);
			date_num(f) = datenum(date_str{f});
		otherwise
			date_num(f) = str2num(files(f).name(end-17));
		end;

	end;

	subs = unique(SN);
	for s = 1:length(subs);
		ind = find(SN==subs(s));
		dates = date_num(ind);
		ordered_days = dates-min(dates)+1;
		session(ind) = ordered_days;
		n_sessions(s) = length(ind);
	end;

	max_sess = max(n_sessions);
	for s = 1:length(subs);
		for sess = 1:n_sessions(s);
			fprintf(sprintf('Group %s Subject %d Session %d\n', group_ID, subs(s), sess))
			ind = find(SN == subs(s) & session == sess);

			[out, Params] = SymbSeq_analysis(fullfile(data_dir,files(ind).name));
			for b = 1:length(out.Block);
				D.Group(data_indx) = c;
				D.SN(data_indx) = subs(s);
				D.Session(data_indx) = session(sess);
				D.Block(data_indx) = b;
				D.MeanRT(data_indx) = out.MeanRT(b);
				D.StdRT(data_indx) = out.StdRT(b);
				D.Acc(data_indx) = out.Acc(b);

				for l = 1:10;
					eval(sprintf('D.Lag%d(data_indx) = out.Lag%d(b);',l,l));
				end;

				data_indx = data_indx + 1;
			end;

			DELTA.Group(sess_indx) = c;
			DELTA.SN(sess_indx)    = subs(s);
			DELTA.Session(sess_indx) = session(sess);
			DELTA.meanRT(sess_indx) = out.MeanRT(5) - mean(out.MeanRT([4 6]));
			DELTA.stdRT(sess_indx)  = mean(out.StdRT([4 6]))-out.StdRT(5);
			DELTA.Acc(sess_indx)    =  mean(out.Acc([4 6]))-out.Acc(5);

			REMAP.Group(sess_indx) = c;
			REMAP.SN(sess_indx) = subs(s);
			REMAP.Session(sess_indx) = session(sess);
			REMAP.On(sess_indx) = Params.REMAP;
			REMAP.Repeats(sess_indx)=length(find(~diff([Params.Seq Params.Seq(1)])));
			sess_indx = sess_indx + 1;
		end;
	end;
	clear SN date_str date_num session n_sessions files;
end;


% For anova replace subject IDs in a sorted way
subs = unique(DELTA.SN);
sub_num = 1:length(subs);

for s = sub_num;
	ind = find(DELTA.SN == subs(s));
	DELTA.orderedSN(ind) = sub_num(s);

	ind = find(D.SN == subs(s));
	D.orderedSN(ind) = sub_num(s);
end;

D = flipD(D);
DELTA = flipD(DELTA);
REMAP = flipD(REMAP)

save('summary_file.mat');

end;

if ~REPROCESS
	load('summary_file.mat');
end;

% Now do the plotting

% Figure 1
figure;
sub_indx = 1;
color_list = flipdim(0:(1/6):0.8,2);
style_list = {'--','-','--','-'};

for c = 1:length(groups);
	group_ind = find(D.Group == c);

	muRT = pivottable(D.Session(group_ind), D.Block(group_ind), D.MeanRT(group_ind), 'mean');
	stdRT = pivottable(D.Session(group_ind), D.Block(group_ind), D.MeanRT(group_ind), 'std');
	semRT = stdRT./sqrt(length(unique(D.SN(group_ind))));

	muAcc = pivottable(D.Session(group_ind), D.Block(group_ind), D.Acc(group_ind), 'mean');
	stdAcc = pivottable(D.Session(group_ind), D.Block(group_ind), D.Acc(group_ind), 'std');
	semAcc = stdAcc./sqrt(length(unique(D.SN(group_ind))));

	% Plot the mean RT results
	subplot(length(groups),2, sub_indx);
	h = errorbar(muRT', semRT','.-');
	for sess = 1:length(h);
		set(h(sess),'Color',color_list(sess).*[1 1 1],'linewidth',2,'MarkerEdgeColor','k');
		ch = get(h(sess),'Children');
		set(ch(2),'Color','k','linewidth',1);
	end;
	set(gca,'XTick',1:6, 'Ylim',[0 600],'YTick',0:100:600, 'Xlim',[0 7]);
	ylabel('RT (ms)');
	xlabel('Blocks');
	drawlines([1:5]+0.5,'k');
	title(group_labels{c});

	sub_indx = sub_indx + 1;

	% Plot Accuracy
	subplot(length(groups),2, sub_indx);
	h = errorbar(muAcc', semAcc','.-');
	for sess = 1:length(h);
		set(h(sess),'Color',color_list(sess).*[1 1 1],'linewidth',2,'MarkerEdgeColor','k');
		ch = get(h(sess),'Children');
		set(ch(2),'Color','k','linewidth',1);
	end;
	set(gca,'XTick',1:6, 'Ylim',[0 1],'YTick',0:.1:1, 'Xlim',[0 7]);
	ylabel('Accuracy');
	xlabel('Blocks');
	drawlines([1:5]+0.5,'k');
	title(group_labels{c});

	sub_indx = sub_indx + 1;

	% N subjects
	N_subs(c) = length(unique(D.SN(group_ind)));
end;
set(gcf,'Position',[452    20   586   785])

% -------------------------------------------------
% Try redoing learning analysis normalizing against Day 1 scores
% -------------------------------------------------
[FA, RA, CA] = pivottable([DELTA.Group DELTA.orderedSN],DELTA.Session,DELTA.meanRT,'mean');

norm_dRT.Group = repmat(RA(:,1)',1,size(FA,2)-1)';
norm_dRT.SN = repmat(RA(:,2)',1,size(FA,2)-1)';
norm_dRT.Session = [ones(size(RA,1),1); 2*ones(size(RA,1),1); 3*ones(size(RA,1),1); 4*ones(size(RA,1),1)];
tmp = FA(:,2:end)-repmat(FA(:,1),1,size(FA,2)-1);
norm_dRT.meanRT_learn = tmp(:);

[FA, RA, CA] = pivottable([DELTA.Group DELTA.SN],DELTA.Session,DELTA.stdRT,'mean');
tmp = FA(:,2:end)-repmat(FA(:,1),1,size(FA,2)-1);
norm_dRT.stdRT_learn = tmp(:);


[FA, RA, CA] = pivottable([DELTA.Group DELTA.SN],DELTA.Session,DELTA.Acc,'mean');
tmp = FA(:,2:end)-repmat(FA(:,1),1,size(FA,2)-1);
norm_dRT.Acc_learn = tmp(:);

figure;
% ----------------------------------------------------------
subplot(3,2,1); hold on;
day1 = find(DELTA.Session==1);
mu = pivottable([], DELTA.Group(day1), DELTA.meanRT(day1),'mean');
sem = pivottable([], DELTA.Group(day1), DELTA.meanRT(day1),'std')./sqrt(15);
for g = 1:4
	h=bar(g,mu(g),'w');
	set(h,'facecolor',color_list(g)*[1 1 1]);
end;
h = errorbar(1:4, mu, sem, 'k.')
set(h,'Color',color_list(g).*[1 1 1],'linewidth',3,'MarkerEdgeColor','k');
set(gca,'XTick',1:4,'XTickLabel',group_labels)
set(gca,'XLim',[0.5 4.5])
ylabel('s\deltaRT');
axis square;

X = pivottable(DELTA.SN, DELTA.Group(day1), DELTA.meanRT(day1),'mean');
[p, ANOVA_TABLE] = anova1(X,group_labels,'off');
Day1_stats{1} = {ANOVA_TABLE{2,5} ANOVA_TABLE{2,6} ANOVA_TABLE{2,3} ANOVA_TABLE{end,3}}


% ----------------------------------------------------------
subplot(3,2,2);
mu = pivottable(norm_dRT.Group, norm_dRT.Session, norm_dRT.meanRT_learn,'mean');
sem = pivottable(norm_dRT.Group, norm_dRT.Session, norm_dRT.meanRT_learn,'std')./sqrt(15);
h = errorbar(mu', sem','.-');
for g = 1:length(h);
	set(h(g),'Color',color_list(g).*[1 1 1],'linewidth',3,'MarkerEdgeColor','k');
	set(h(g),'linestyle',style_list{g})
	ch = get(h(g),'Children');
	set(ch(2),'Color','k','linewidth',1);
end;
ylabel('\Delta_\deltaRT');
set(gca,'XTick',1:4,'XTickLabel',2:5,'YTick',-25:25:200);
set(gca,'XLim',[0.5 4.5])
xlabel('Day');
axis square;
legend(group_labels)

% ----------------------------------------------------------
subplot(3,2,3); hold on;
mu = pivottable([], DELTA.Group(day1), DELTA.Acc(day1),'mean');
sem = pivottable([], DELTA.Group(day1), DELTA.Acc(day1),'std')./sqrt(15);
for g = 1:4
	h=bar(g,mu(g),'w');
	set(h,'facecolor',color_list(g)*[1 1 1]);
end;
h = errorbar(1:4, mu, sem, 'k.')
set(h,'Color',color_list(g).*[1 1 1],'linewidth',3,'MarkerEdgeColor','k');
set(gca,'XTick',1:4,'XTickLabel',group_labels)
set(gca,'XLim',[0.5 4.5])
ylabel('\deltaAcc');
axis square;

X = pivottable(DELTA.SN, DELTA.Group(day1), DELTA.Acc(day1),'mean');
[p, ANOVA_TABLE] = anova1(X,group_labels,'off');
Day1_stats{2} = {ANOVA_TABLE{2,5} ANOVA_TABLE{2,6} ANOVA_TABLE{2,3} ANOVA_TABLE{end,3}}

% ----------------------------------------------------------
subplot(3,2,4);
mu = pivottable(norm_dRT.Group, norm_dRT.Session, norm_dRT.Acc_learn,'mean');
sem = pivottable(norm_dRT.Group, norm_dRT.Session, norm_dRT.Acc_learn,'std')./sqrt(15);
h = errorbar(mu', sem','.-');
for g = 1:length(h);
	set(h(g),'Color',color_list(g).*[1 1 1],'linewidth',3,'MarkerEdgeColor','k');
	set(h(g),'linestyle',style_list{g})
	ch = get(h(g),'Children');
	set(ch(2),'Color','k','linewidth',1);
end;
ylabel('\Delta_\deltaAcc');
set(gca,'XTick',1:4,'XTickLabel',2:5,'YTick',-0.05:0.05:0.25);
set(gca,'XLim',[0.5 4.5])
xlabel('Day');
axis square;

% ----------------------------------------------------------
subplot(3,2,5); hold on;
mu = pivottable([], DELTA.Group(day1), DELTA.stdRT(day1),'mean');
sem = pivottable([], DELTA.Group(day1), DELTA.stdRT(day1),'std')./sqrt(15);
for g = 1:4
	h=bar(g,mu(g),'w');
	set(h,'facecolor',color_list(g)*[1 1 1]);
end;
h = errorbar(1:4, mu, sem, 'k.')
set(h,'Color',color_list(g).*[1 1 1],'linewidth',3,'MarkerEdgeColor','k');
set(gca,'XTick',1:4,'XTickLabel',group_labels)
set(gca,'XLim',[0.5 4.5])
ylabel('\delta\sigma');
axis square;

X = pivottable(DELTA.SN, DELTA.Group(day1), DELTA.stdRT(day1),'mean');
[p, ANOVA_TABLE] = anova1(X,group_labels,'off');
Day1_stats{3} = {ANOVA_TABLE{2,5} ANOVA_TABLE{2,6} ANOVA_TABLE{2,3} ANOVA_TABLE{end,3}}

% ----------------------------------------------------------
subplot(3,2,6);
mu = pivottable(norm_dRT.Group, norm_dRT.Session, norm_dRT.stdRT_learn,'mean');
sem = pivottable(norm_dRT.Group, norm_dRT.Session, norm_dRT.stdRT_learn,'std')./sqrt(15);
h = errorbar(mu', sem','.-');
for g = 1:length(h);
	set(h(g),'Color',color_list(g).*[1 1 1],'linewidth',3,'MarkerEdgeColor','k');
	set(h(g),'linestyle',style_list{g})
	ch = get(h(g),'Children');
	set(ch(2),'Color','k','linewidth',1);
end;
ylabel('\Delta_\delta\sigma');
set(gca,'XTick',1:4,'XTickLabel',2:5,'YTick',-10:10:40);
set(gca,'XLim',[0.5 4.5])
xlabel('Day');
axis square;


[~, DFs, ~, Fs, Ps] = mixed_between_within_anova([norm_dRT.meanRT_learn norm_dRT.Group norm_dRT.Session norm_dRT.SN]);
normRT_stats{1} = {Fs, DFs, Ps};

[~, DFs, ~, Fs, Ps] = mixed_between_within_anova([norm_dRT.Acc_learn norm_dRT.Group norm_dRT.Session norm_dRT.SN]);
normRT_stats{2} = {Fs, DFs, Ps};

[~, DFs, ~, Fs, Ps] = mixed_between_within_anova([norm_dRT.stdRT_learn norm_dRT.Group norm_dRT.Session norm_dRT.SN]);
normRT_stats{3} = {Fs, DFs, Ps};

set(gcf,'position',[92 8 565 797])


% Evaluate Day by Day effects
for d = 1:4;
	d_indx = find(norm_dRT.Session==d);
	X = pivottable(repmat([1:15]',4,1),norm_dRT.Group(d_indx), norm_dRT.meanRT_learn(d_indx),'mean');
	[~, foo] = anova1(X, group_labels,'off');
	eval(sprintf('RT_day%d_Ftable = foo;',d+1));

	X = pivottable(repmat([1:15]',4,1),norm_dRT.Group(d_indx), norm_dRT.Acc_learn(d_indx),'mean');
	[~, foo] = anova1(X, group_labels,'off');
	eval(sprintf('Acc_day%d_Ftable = foo;',d+1));
end;




% -------------------------------------------------
% Show ato-correlation
% -------------------------------------------------
N = length(unique(D.SN));

% Calculate the binding coefficiencts
% Replot figure 3
for lag = 1:10

	ind = find(D.Block == 5);
	eval(sprintf('lag_data = D.Lag%d(ind);', lag));
	B5(:,:,lag) = pivottable(D.SN(ind),D.Session(ind),lag_data,'mean');

	ind = find(D.Block == 6);
	eval(sprintf('lag_data = D.Lag%d(ind);', lag));
	B6(:,:,lag) = pivottable(D.SN(ind),D.Session(ind),lag_data,'mean');
end;

color_list = flipdim(0:(1/6):0.8,2);
figure;
subplot(1,2,1);
h = errorbar(squeeze(mean(B5))', squeeze(std(B5))' ./ sqrt(N), 'k.-');
for g = 1:length(h);
	set(h(g),'Color',color_list(g).*[1 1 1],'linewidth',3,'MarkerEdgeColor','k');
	ch = get(h(g),'Children');
	set(ch(2),'Color','k','linewidth',1);
end;
legend('Day 1','Day 2','Day 3','Day 4', 'Day 5')
title('Block 5 (Random)')
set(gca,'Ylim',[-0.1 0.3],'YTick',-0.1:.1:.3)
set(gca,'Xlim',[0 11],'XTick',1:10)
ylabel('Correlation'); xlabel('Lag (trials)')
axis square

subplot(1,2,2);
h = errorbar(squeeze(mean(B6))', squeeze(std(B6))' ./ sqrt(N), 'k.-');
for g = 1:length(h);
	set(h(g),'Color',color_list(g).*[1 1 1],'linewidth',3,'MarkerEdgeColor','k');
	ch = get(h(g),'Children');
	set(ch(2),'Color','k','linewidth',1);
end;
legend('Day 1','Day 2','Day 3','Day 4', 'Day 5')
title('Block 6 (Sequence)')
set(gca,'Ylim',[-0.1 0.3],'YTick',-0.1:.1:.3)
set(gca,'Xlim',[0 11],'XTick',1:10)
ylabel('Correlation'); xlabel('Lag (trials)')
axis square


% -------------------------------------------------
% Estimate bindign as 3-lag diff
% -------------------------------------------------
n_lags = 3;
clear foo tmp
sesses = [1 5]
Y = []; G =[]; S = []; DY = [];
for g = 1:length(groups);
	for sess = 1:length(sesses)
		ind = find(D.Group == g & D.Session == sesses(sess));

		for lag = 1:n_lags
			eval(sprintf('lag_data = D.Lag%d(ind);', lag));
			foo  = pivottable(D.SN(ind),D.Block(ind),lag_data,'mean');

			tmp(:,lag) = foo(:,6)-foo(:,5);
		end;

		bind_score = squeeze(mean(tmp,2));

		Y = [Y; bind_score];
		G = [G; repmat(g,size(foo,1),1)];
		DY = [DY; repmat(sess,size(foo,1),1)];
		S = [S; unique(D.orderedSN(ind))];
	end;
end;


%binding_score_stats = rm_anova2(Y, S, G, DY, {'Group','Session'})
[~, DFs, ~, Fs, Ps] = mixed_between_within_anova([Y G DY S]);
binding_score_stats = {Fs, Ps, DFs};

seq_binding = reshape(Y,15,length(sesses),length(groups));
[~, d1_bind_stats] = anova1(squeeze(seq_binding(:,1,:)),group_labels,'off');
[~, d5_bind_stats] = anova1(squeeze(seq_binding(:,end,:)),group_labels,'off');

figure;
d1 = squeeze(seq_binding(:,1,:));
d5 = squeeze(seq_binding(:,end,:));

bar([mean(d1); mean(d5)]'); hold on;
A = colormap('gray');
colormap(flipdim(A,1));


h = errorbar([1:4]-0.15,mean(d1),std(d1)./sqrt(15),'k.');hold on;
set(h,'linewidth',1,'markersize',1)
h = errorbar([1:4]+0.15,mean(d5),std(d5)./sqrt(15),'k.');hold on;
set(h,'linewidth',1,'markersize',1)

set(gca,'XTick',1:4,'XTickLabel',group_labels)
ylabel('Binding Score');
legend('Day 1','Day 5')

d_seq_binding = squeeze(seq_binding(:,end,:)-seq_binding(:,1,:));
t_seq_binding = mean(d_seq_binding)./(std(d_seq_binding)./sqrt(size(d_seq_binding,1)));
p_seq_binding = 1-tcdf(t_seq_binding,size(d_seq_binding,1)-1);

dY = [];
for g = 1:4;
	d5_indx = find(G == g & DY == 2);
	d1_indx = find(G == g & DY == 1);

	dY = [dY; Y(d5_indx) - Y(d1_indx)];
end;

foo = reshape(G,2,60);
X = [zeros(size(dY,1), 3)];
for c = 1:3;
	X(find(foo(1,:)==c),c) = 1;
end;


[~, ~, ~, ~, betas] = bootstrap_regression(X, dY,[1 0 0]);
y_hat = [X ones(60,1)]*betas;
SSM = sum([y_hat - mean(y_hat)].^2);
SSE = sum([dY(:) - y_hat].^2); 
slope_F = (SSM/1)/(SSE/(length(y_hat)-2));
slope_p_F = 1-fcdf(slope_F, 1, length(dY(:))-2);

% -------------------------------------------------
% Estiamte across day learning scores
% -------------------------------------------------

[FA, RA] = pivottable([DELTA.Group DELTA.SN], DELTA.Session, DELTA.meanRT,'mean');

slope = reshape(mean(diff(FA,[],2),2),15,4);
mu_slope = mean(slope);
sem_slope = std(slope)./sqrt(15);
t_slope = mu_slope./sem_slope;

X = zeros(size(RA,1), 3);
for c = 1:3;
	X(find(RA(:,1)==c),c) = 1;
end;

[~, ~, ~, ~, betas] = bootstrap_regression(X, slope(:),[1 0 0]);
y_hat = [X ones(60,1)]*betas;
SSM = sum([y_hat - mean(y_hat)].^2);
SSE = sum([slope(:) - y_hat].^2); 
F(1) = (SSM/1)/(SSE/(length(y_hat)-2));
c_F(1) = 1-fcdf(F(1), 1, length(slope(:))-2);

for g1 = 1:4;
	for g2 = 1:4;
		[RT_t(g1,g2), RT_p(g1,g2)] = ttest(slope(:,g1), slope(:,g2),2,'independent');
	end;
end;


figure;
subplot(1,2,1); 
hold on
for g = 1:4
	h=bar(g,mu_slope(g),'w');
	set(h,'facecolor',color_list(g)*[1 1 1]);
end;
h = errorbar(1:4, mu_slope, sem_slope, 'k.')
set(h,'Color',color_list(g).*[1 1 1],'linewidth',3,'MarkerEdgeColor','k');
set(gca,'XTick',1:4,'XTickLabel',group_labels)
set(gca,'XLim',[0.5 4.5])
ylabel('\Delta_R_T');
axis square;


[FA, RA] = pivottable([DELTA.Group DELTA.SN], DELTA.Session, DELTA.Acc,'mean');

slope = reshape(mean(diff(FA,[],2),2),15,4);
mu_slope = mean(slope);
sem_slope = std(slope)./sqrt(15);
t_slope = mu_slope./sem_slope;

X = zeros(size(RA,1), 3);
for c = 1:3;
	X(find(RA(:,1)==c),c) = 1;
end;

[~, ~, ~, ~, betas] = bootstrap_regression(X, slope(:),[1 0 0]);
y_hat = [X ones(60,1)]*betas;
SSM = sum([y_hat - mean(y_hat)].^2);
SSE = sum([slope(:) - y_hat].^2); 
F(2) = (SSM/1)/(SSE/(length(y_hat)-2));
c_F(2) = 1-fcdf(F(2), 1, length(slope(:))-2);

subplot(1,2,2); 
hold on
for g = 1:4
	h=bar(g,mu_slope(g),'w');
	set(h,'facecolor',color_list(g)*[1 1 1]);
end;
h = errorbar(1:4, mu_slope, sem_slope, 'k.')
set(h,'Color',color_list(g).*[1 1 1],'linewidth',3,'MarkerEdgeColor','k');
set(gca,'XTick',1:4,'XTickLabel',group_labels)
set(gca,'XLim',[0.5 4.5])
ylabel('\Delta_A_c_c');
axis square;


for g1 = 1:4;
	for g2 = 1:4;
		[Acc_t(g1,g2), Acc_p(g1,g2)] = ttest(slope(:,g1), slope(:,g2),2,'independent');
	end;
end;

set(gcf,'position',[440   467   799   331]);
set(gca,'ylim',[0 0.06])

return;

%subplot(2,2,4);
figure;
h = errorbar(squeeze(mean(seq_binding)), squeeze(std(seq_binding)) ./ sqrt(15), 'k.-');
for g = 1:length(h);
	set(h(g),'Color',color_list(g).*[1 1 1],'linewidth',3,'MarkerEdgeColor','k','markersize',0.5);
	set(h(g),'linestyle',style_list{g})
	ch = get(h(g),'Children');
	set(ch(2),'Color','k','linewidth',1);
end;
xlabel('Training Session')
set(gca,'Ylim',[-0.1 0.3],'YTick',-0.1:.1:.3)
set(gca,'XTick',1:5)
ylabel('Binding Score');
legend(group_labels)
