function T  = tapply(D,categories,varargin) 
% T  = tapply(D,categories,{dependent 1},{dependent 2},...)
% Condenses a data structure into a new data structure 
% EXAMPLE:
%   tapply(D,{'Cond','Subcond'},{'RT','mean',subset',D.SN == 1,'name','mRT'},...,'subset',D.good)
%   makes a new data frame with variables Cond, Subcond 
%   and other variables that are calculated based on prescription
% --------------------------------------------------------------
% Joern Diedrichsen jdiedric@jhu.edu
Dep={};R=[];T=[];
if (~iscell(categories))
    categories={categories};
end;
for i=1:length(categories)
    R=[R getfield(D,categories{i})]; 
end;
[rR,cR]=size(R);
index=logical(ones(size(R,1),1));
c=1;
while c<=length(varargin)
    if (iscell(varargin{c}))
        Dep{end+1}=varargin{c};
        c=c+1;
    else 
        switch (varargin{c})
            case 'subset'
                index=varargin{c+1};c=c+2;
            otherwise 
                error('tapply: unknown option');
        end;
    end;
end;



% Now do all the pivottables for all the Y
TR=[];FR=[];dv=1;
for dvar=1:length(Dep)
    [F,Cat,name{dvar}]=do_dependent(Dep{dvar},R,D,index);
    FR=[FR ones(size(FR,1),1)*NaN];
    for i=1:size(Cat,1)
        x=findrow(TR,Cat(i,:));
        if (isempty(x))
            TR(end+1,:)=Cat(i,:);
            x=size(TR,1);
            FR(end+1,:)=ones(1,size(FR,2))*NaN;
        end;
        FR(x,dv:dv+size(F,2)-1)=F(i,:);
    end;
    dv=dv+size(F,2);

end;

% Now tranfer the fields
for c=1:length(categories)
    T=setfield(T,categories{c},TR(:,c));
end;
dvar=1;
for c=1:length(name)
    if (iscell(name{c}))
        for v=1:length(name{c})
            T=setfield(T,name{c}{v},FR(:,dvar));
            dvar=dvar+1;    
        end;
    else
        T=setfield(T,name{c},FR(:,dvar));
        dvar=dvar+1;
    end;
end;
% keyboard;


function [F,Cat,name]=do_dependent(Dep,R,D,index);
if ischar(Dep{1})
    name=Dep{1};
    Y=getfield(D,name);
else 
    name='var';
    Y=Dep{1};
end;
if length(Dep)<2
    fcn='mean';
else
    fcn=Dep{2};
end;
c=3;
Column=[];
while c<=length(Dep)
    switch (Dep{c})
        case 'name'
            name=Dep{c+1};
            c=c+2;
        case 'subset'
            index=index & Dep{c+1};
            c=c+2;
        case 'split'
            Column=Dep{c+1};
            c=c+2;
    end;
end;
[F,Cat,Split]=pivottable(R,Column,Y,fcn,'subset',index);
if (~isempty(Split))
    for i=1:size(Split,1)
        New_name{i}=[name sprintf('_%d',Split(i,:))];
    end;
    name=New_name;
end;
    