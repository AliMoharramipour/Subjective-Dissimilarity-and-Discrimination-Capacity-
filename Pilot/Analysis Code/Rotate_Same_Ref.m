function Y=Rotate_Same_Ref(Y)


ThetaRotate=pi-atan(Y(1,2)/Y(1,1));
A=[cos(ThetaRotate) -sin(ThetaRotate) 0;sin(ThetaRotate) cos(ThetaRotate) 0;0 0 1];
Y=A*[Y';ones(1,size(Y,1))];
Y=Y(1:2,:)';
if(Y(2,2)<0)
    Y(:,2)=Y(:,2)*-1;
end
if(Y(1,1)>0)
    Y(:,1)=Y(:,1)*-1;
end


