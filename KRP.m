% Function KRP - Evaluates the Khatri-Rao Product 
% Inputs : A and B
% Outputs : C - the Khatri-Rao Product

function C = KRP(A, B)
C=[];
for i = 1: size(A,2)
    pr = (kron(A(:,i), B(:,i)));
    C = [C pr(:)];
end
end