function [indx,R]=putcat(r,R)
% function [indx,R]=putcat(r,R)
% puts the r into R, if not present, otherwise return index
% r: can be a row vector 
% R: has to have as many variables (columns) as r, and has as many rows as 
% unique categories present
[numrows,numcols]=size(R);
found=0;
% empty Matrix: return 1 
if (isempty(r))
      indx=1;
      R=NaN;
      return;
  end;    
  for indx=1:numrows  %go through all rows in R 
     issame=(r==R(indx,:));
     issame(find(isnan(r) & isnan(R(indx,:))))=1;
     if (issame) %find row with this entree
        found=indx;
     end; 
  end;
  if(found==0)
     R=[R;r];
     [indx,dummy]=size(R);
  else 
     indx=found;
  end;
