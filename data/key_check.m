close all;
clear all;


group_ID = 'A';

switch group_ID
	case {'A','B'};
		files = dir(fullfile(sprintf('*%s-*SymbSeq*.mat',group_ID)));
	otherwise
		files = dir(fullfile(sprintf('*_%s_*SymbSeq*.mat',group_ID)));
end;



for f = 1:length(files)
    
    load(files(f).name);
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
    
    ind = find(BlockNum == 6);
    
    for t = 1:length(ind)
        K(t) = find(strcmpi(Params.Key,Data(t).TargetKey));
        C(t) = Data(t).CueType;
    end;
    
    match_prob(f) = length(find(abs(K-C)));
    SN(f) = str2num(files(f).name(1:3));
    Remap(f) = Params.REMAP;
    KeyMap(f,:) = Params.key_map;
    
    switch group_ID
        case 'C'
            Session(f) = str2num(files(f).name(8));
        otherwise
            Session(f) = datenum(files(f).name(7:17));
    end;
    
    
    clear Data Params;
end;