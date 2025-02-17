%%%%%%%%%%%%%%%%%% Select Pairs for the discrimination task %%%%%%%%%%%%%%%%%%%
clc;
clear;
rng("default")

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DissimCalculateApproach='MDS_5d'; %%% 'ML','MDS_5d','MDS_2d'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SubjIDs={'415090147-1','415090147-2',...
         '619502508-1','619502508-2',...
         '089858508-1','089858508-2',...
         '951506687-1','951506687-2',...
         '584740030-1','584740030-2',...
         '518402380-1','518402380-2',...
         '801165888-1','801165888-2',...
         '940332894-1','940332894-2',...
         '682207766-1','682207766-2',...
         '040062257-1','040062257-2',...
         '695758433-1','695758433-2',...
         '108169884-1','108169884-2'};

Reps=[1 2;3 4;5 6;7 8;9 10;11 12;13 14;15 16;17 18;19 20;21 22;23 24]; %%% Same subject data index %%%

L1L2=repmat([0.00025 0.00025],24,1); %%% L1,L2 hyperparameters of the ML approach


%%%%%%%% Data Address %%%%%%
CurrentAddress=pwd;
S=strsplit(CurrentAddress, filesep);
S(end)=[];
Data_Sim_Address=[];
for i=1:length(S)
    Data_Sim_Address=[Data_Sim_Address S{i} filesep];
end
Data_Sim_Address=[Data_Sim_Address 'Collected Data' filesep 'Subjective Similarity Judgment Task'];


%%%%%************ Load Dissim matrices ************%%%%%%
Dissim=[];
for i=1:size(Reps,1)

    Dissim_o_in_each_Sess=[];
    Dissim_from_embedding_5d_in_each_sess=[];
    Dissim_ml_in_each_Sess=[];
    disp('************* One Subj ******************')
    disp('## Dissimilarity ##')
    for n=1:length(Reps(i,:))
        %%%% Read the two sessions of the same participants %%%%%
        Files=dir(Data_Sim_Address);
        for j=1:length(Files)
            Check=strsplit(Files(j).name,SubjIDs{Reps(i,n)});
            if(length(Check)>1)
                break;
            end
        end

        disp(Files(j).name)
        load([Data_Sim_Address filesep Files(j).name filesep 'dissim_matrix.csv'])
        Dissim_o_in_each_Sess{n}=dissim_matrix; %%% original dissim matrix based on response probabilities %%%%
        
        %%%%% 5d MDS applied to the above original dissim matrix %%%%%
        D=Dissim_o_in_each_Sess{n};
        D(D==0)=nan;
        D(eye(size(D))==1)=0;
        D=dist(mdscale(D,5,'Criterion','metricstress','Start','random')');
        Dissim_from_embedding_5d_in_each_sess{n}=D/max(D(:));

        %%%%% load the embeddings obtained by the ML approach (python code) %%%%%%
        ML_Approach_Embeddings=readtable([Data_Sim_Address filesep Files(j).name filesep 'ML_Approach' filesep 'Space_' num2str(L1L2(Reps(i,n),1)) '_' num2str(L1L2(Reps(i,n),2)) '.csv']);
        ML_Approach_Embeddings=table2array(ML_Approach_Embeddings(2:end,:));
        Std_Embeddings=std(ML_Approach_Embeddings); 
        Std_Embeddings_Percent=100*Std_Embeddings/sum(Std_Embeddings);
        Dissim_ml_in_each_Sess{n}=dist(ML_Approach_Embeddings');
        Dissim_ml_in_each_Sess{n}=Dissim_ml_in_each_Sess{n}/max(Dissim_ml_in_each_Sess{n}(:));
    end

    %%%%%% Average the dissim matrices from the two sessions %%%%%
    switch DissimCalculateApproach
        case 'ML'
            D1=Dissim_ml_in_each_Sess{1};
            D2=Dissim_ml_in_each_Sess{2};
            Dissim{i}=(D1+D2)/2;

        case 'MDS_5d'
            D1=Dissim_from_embedding_5d_in_each_sess{1};
            D2=Dissim_from_embedding_5d_in_each_sess{2};
            Dissim{i}=(D1+D2)/2;


        case 'MDS_2d'
            D1=Dissim_o_in_each_Sess{1};
            D1(D1==0)=nan;
            D1(eye(size(D1))==1)=0;
            D1=dist(mdscale(D1,2,'Criterion','metricstress','Start','random')')';
            D1=D1/max(D1(:));

            D2=Dissim_o_in_each_Sess{2};
            D2(D2==0)=nan;
            D2(eye(size(D2))==1)=0;
            D2=dist(mdscale(D2,2,'Criterion','metricstress','Start','random')')';
            D2=D2/max(D2(:));

            Dissim{i}=(D1+D2)/2;
    end
end

%%%%%%**************** each pairs dissimilarity stats accross participants ********************%%%%%%%%
PickUpperTriangle=triu(ones(30,30));
PickUpperTriangle(eye(30)==1)=0;
PickUpperTriangle=find(PickUpperTriangle);
PairIndices=[];
[PairIndices(:,1),PairIndices(:,2)]=ind2sub([30 30],PickUpperTriangle);
%%% z_noramlize %%%
for i=1:length(Dissim)
    M=mean(Dissim{i}(PickUpperTriangle));
    S=std(Dissim{i}(PickUpperTriangle));
    Dissim{i}=(Dissim{i}-M)/S;
end

%%% Median and MAD accross subjects %%%%%
All_Dissim_z=[];
for i=1:length(Dissim)
    All_Dissim_z(:,:,i)=Dissim{i};
end
Median_Dissim=median(All_Dissim_z,3);
MAD_Dissim=median(abs(All_Dissim_z-repmat(median(All_Dissim_z,3),1,1,size(All_Dissim_z,3))),3);
Group_Median_Pairs=Median_Dissim(PickUpperTriangle);
Group_MAD_Pairs=MAD_Dissim(PickUpperTriangle);

%%%%%% first Quantile %%%%
Q1_lim=quantile(Group_Median_Pairs,0.25);
Q1_Ind=find(Group_Median_Pairs<=Q1_lim);

[Group_MAD_Pairs_Q1_sorted,ind]=sort(Group_MAD_Pairs(Q1_Ind));
ind=flipud(ind);
Group_MAD_Pairs_Q1_sorted=flipud(Group_MAD_Pairs_Q1_sorted);

Q1_Ind=Q1_Ind(ind);
Q1_Pair_Candidates=array2table([PairIndices(Q1_Ind,:) Group_Median_Pairs(Q1_Ind) Group_MAD_Pairs(Q1_Ind)],'VariableNames',{'Face1', 'Face2', 'GroupMedian','GroupMAD'});

disp('################################# Q1 ##################################')
disp('************** Top 20 *************')
disp(Q1_Pair_Candidates(1:20,:));
disp('************** Bottom 10 *************')
disp(Q1_Pair_Candidates(end-9:end,:));

%%%%%% second Quantile %%%%
Q2_lim=quantile(Group_Median_Pairs,0.5);
Q2_Ind=find(Group_Median_Pairs>Q1_lim & Group_Median_Pairs<=Q2_lim);

[Group_MAD_Pairs_Q2_sorted,ind]=sort(Group_MAD_Pairs(Q2_Ind));
ind=flipud(ind);
Group_MAD_Pairs_Q2_sorted=flipud(Group_MAD_Pairs_Q2_sorted);

Q2_Ind=Q2_Ind(ind);
Q2_Pair_Candidates=array2table([PairIndices(Q2_Ind,:) Group_Median_Pairs(Q2_Ind) Group_MAD_Pairs(Q2_Ind)],'VariableNames',{'Face1', 'Face2', 'GroupMedian','GroupMAD'});

disp('################################# Q2 ##################################')
disp('************** Top 20 *************')
disp(Q2_Pair_Candidates(1:20,:));
disp('************** Bottom 10 *************')
disp(Q2_Pair_Candidates(end-9:end,:));

%%%%%% Third Quantile %%%%
Q3_lim=quantile(Group_Median_Pairs,0.75);
Q3_Ind=find(Group_Median_Pairs>Q2_lim & Group_Median_Pairs<=Q3_lim);

[Group_MAD_Pairs_Q3_sorted,ind]=sort(Group_MAD_Pairs(Q3_Ind));
ind=flipud(ind);
Group_MAD_Pairs_Q3_sorted=flipud(Group_MAD_Pairs_Q3_sorted);

Q3_Ind=Q3_Ind(ind);
Q3_Pair_Candidates=array2table([PairIndices(Q3_Ind,:) Group_Median_Pairs(Q3_Ind) Group_MAD_Pairs(Q3_Ind)],'VariableNames',{'Face1', 'Face2', 'GroupMedian','GroupMAD'});

disp('################################# Q3 ##################################')
disp('************** Top 20 *************')
disp(Q3_Pair_Candidates(1:20,:));
disp('************** Bottom 10 *************')
disp(Q3_Pair_Candidates(end-9:end,:));

%%%%%% Forth Quantile %%%%
Q4_Ind=find(Group_Median_Pairs>Q3_lim);

[Group_MAD_Pairs_Q4_sorted,ind]=sort(Group_MAD_Pairs(Q4_Ind));
ind=flipud(ind);
Group_MAD_Pairs_Q4_sorted=flipud(Group_MAD_Pairs_Q4_sorted);

Q4_Ind=Q4_Ind(ind);
Q4_Pair_Candidates=array2table([PairIndices(Q4_Ind,:) Group_Median_Pairs(Q4_Ind) Group_MAD_Pairs(Q4_Ind)],'VariableNames',{'Face1', 'Face2', 'GroupMedian','GroupMAD'});

disp('################################# Q4 ##################################')
disp('************** Top 20 *************')
disp(Q4_Pair_Candidates(1:20,:));
disp('************** Bottom 10 *************')
disp(Q4_Pair_Candidates(end-9:end,:));

