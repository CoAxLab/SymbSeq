function fD = flipD(D);

fields = fieldnames(D);

for f = 1:length(fields);
    
    fstr = fields{f};
    
    eval(sprintf('fD.%s = transpose(D.%s);',fstr,fstr));
end;