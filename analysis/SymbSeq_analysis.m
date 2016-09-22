function [O, Params] = SymbSeq_v1p0_analysis(file, DO_PLOT)

if nargin < 2 | isempty(DO_PLOT);
    DO_PLOT = 0;
end;

load(file)
skip_trials = 24;
max_lags = 12;

fields = fieldnames(Data);

for f = 1:length(fields)
	eval(sprintf('%s=[Data.%s];',fields{f},fields{f}));
end;

Keys = zeros(size(CueType));
for k = 1:4;
    indx = find(CueType==k);
    Keys(indx) = Params.key_map(k);
end;

% Filter out missed key presses
miss = find(RT > Params.MaxTrialTime);
RT(miss)=NaN;

% find the set of not prac trials
prac_trials = length(Params.Symbol)*Params.NPracTrialReps;
n_trials = length(Params.Seq)*Params.NTrialReps;

BlockNum = [];
for b = 1:Params.NBlocks-1;

	if b == 1 | b == 2 | b == 5;
		BlockNum = [BlockNum repmat(b,1,n_trials)];
	else;
		BlockNum = [BlockNum repmat(b,1,n_trials)];
	end;
end;


if DO_PLOT
    figure;
    plot(find(~isRand),RT(find(~isRand)),'b.','markersize',15); hold on;
    plot(find(isRand),RT(find(isRand)),'r.','markersize',15)
    %drawlines(find(diff(BlockNum)),'k')
    ylabel('RT (sec)'); xlabel('Trial Num')
end;


for b = 1:max(BlockNum)
	indx = find(BlockNum == b);
	O.Block(b) = b;
	O.MeanRT(b)=nanmean(RT(indx)).*1000;
	O.StdRT(b)= nanstd(RT(indx)).*1000;
	O.Acc(b) = nanmean(Corr(indx));
    
    block_rt = RT(indx);
    good_trials = block_rt(skip_trials:end);
    
    block_cues = CueType(indx);
    good_cues = block_cues(skip_trials:end);

    block_keys = Keys(indx);
    good_keys = block_keys(skip_trials:end);

    for c = 1:4
        eval(sprintf('O.cue%d_RT(b) = nanmean(good_trials(find(good_cues==c)));',c));
        eval(sprintf('O.keys%d_RT(b) = nanmean(good_trials(find(good_keys==c)));',c));
    end;

    [B, INT, R] = regress(good_trials(:),[1:length(good_trials); ones(1,length(good_trials))]');
    
    % Find the nans
    nan_indx = find(isnan(R));
    if ~isempty(nan_indx);
        R(nan_indx) = nanmean(R);
    end;
    
    [C, LAGS] = xcorr(R,max_lags,'coeff');
    
    for l = 1:max_lags
        eval(sprintf('O.Lag%d(b) = C(max_lags+l+1);',l));
    end;

    
end;

O = flipD(O);

[fp, fn, fe] = fileparts(file);
out_file = fullfile(fp,[fn '.dat']);
dsave(out_file,O)
