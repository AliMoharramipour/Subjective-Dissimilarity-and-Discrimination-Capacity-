%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Generate Morphs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear;

CurrentFolderAddress = pwd;
addpath([CurrentFolderAddress filesep 'Code']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Save_Folder=[CurrentFolderAddress filesep 'Morphs'];
Step=1000;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Face1_ID=2;
Face2_ID=10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%% load the FaceSet %%%%%%
load([CurrentFolderAddress filesep 'faces' filesep 'FacesCoordinates.mat']);

%%% Load the model %%%%%=
model = load([CurrentFolderAddress filesep 'Code' filesep '01_MorphableModel.mat']);
msz.n_shape_dim = size(model.shapePC, 2);
msz.n_tex_dim   = size(model.texPC,   2);
msz.n_seg       = size(model.segbin,  2);


%%%%%%%%%
Face1_alpha=alpha(:,Face1_ID).*model.shapeEV;
Face2_alpha=alpha(:,Face2_ID).*model.shapeEV;
Face1_beta=beta(:,Face1_ID).*model.texEV;
Face2_beta=beta(:,Face2_ID).*model.texEV;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
V=Face2_alpha'-Face1_alpha';
Coords_alpha=Face1_alpha'+repmat(V,Step+2,1).*repmat(linspace(0,1,Step+2)',1,msz.n_shape_dim);

V=Face2_beta'-Face1_beta';
Coords_beta=Face1_beta'+repmat(V,Step+2,1).*repmat(linspace(0,1,Step+2)',1,msz.n_tex_dim);

Coords_All=[Coords_alpha Coords_beta];
Face1_All=[Face1_alpha' Face1_beta'];

Distance=[];
for i=1:Step+2
    Distance(i)=norm(Coords_All(i,:)'-Face1_All');
end


Save_Folder=[Save_Folder filesep 'Morphs' num2str(Face1_ID) '-' num2str(Face2_ID)];
mkdir(Save_Folder);
cd(Save_Folder);

for i=2:Step+1

    shape  = coef2object( Coords_alpha(i,:)'./model.shapeEV, model.shapeMU, model.shapePC, model.shapeEV );
    tex    = coef2object( Coords_beta(i,:)'./model.texEV,  model.texMU,   model.texPC,   model.texEV );

    figure
    rp     = defrp;
    rp.phi = 0;
    rp.dir_light.dir = [0;1;1];
    rp.dir_light.intens = 0.6*ones(3,1);
    display_face(shape, tex, model.tl, rp);
    axis off

    print(gcf,[num2str(Face1_ID) '-' num2str(Face2_ID) '.' num2str(i-1) '.png'],'-dpng','-r300');
    if(mod(i,50)==0)
        close all;
    end
        
end
close all;



