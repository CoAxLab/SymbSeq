function T=pivotstruct(D,R_cat,C_cat,variable,fieldcommand,select)

T=[];R=[];C=[];
for i=1:length(R_cat)
    R=[R getfield(D,R_cat{i})]; 
end;
for i=1:length(C_cat)
    C=[C getfield(D,C_cat{i})]; 
end;
[rR,cR]=size(R);
if (nargin<6)
    index=[1:rR]';
else
    index=find(select);
end;
Y=getfield(D,variable);
[Row,Col,Field]=pivottable(R(index,:),C(index,:),Y(index,:),fieldcommand);
for i=1:length(R_cat)
    %T.(categories{i})=Row(:,i);
    T=setfield(T,R_cat{i},Row(:,i));
end;
[Colrow,Colcol]=size(Col);
for i=1:Colcol
    name=variable(1:min(3,length(variable)));
    for r=1:Colrow
        x=sprintf('_%1.0f',Col(r,i));
        name=[name x];
    end;
    T=setfield(T,name,Field(:,i));
end;
