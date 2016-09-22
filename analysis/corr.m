function r=corr(X)
% correlation coefficient 
% function r=corr(X)
% X is a 2-column data vector
% ignores NaN's in one of the vectors
i=find(~isnan(X(:,1)) & ~isnan(X(:,2)));
A=corrcoef(X(i,:));  
if isnan(A)
    r=NaN;
else
    r=A(2,1);
end;
