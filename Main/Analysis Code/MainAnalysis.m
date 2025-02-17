clc;
clear;
rng("default")

%%%%%%%%%%**************** Choose dissimilarity matrix calculation approach **************%%%%%%%%%%%
DissimCalculateApproach='ML'; %%% 'ML','MDS_5d','MDS_2d'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Add BayesFactor Matlab Package %%%
addpath(genpath([pwd filesep 'BayesFactor Matlab Package']));
installBayesFactor

CurrentAddress=pwd;
S=strsplit(CurrentAddress, filesep);
S(end)=[];
Data_Address=[];
for i=1:length(S)
    Data_Address=[Data_Address S{i} filesep];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Data_Sim_Address=[Data_Address 'Collected Data' filesep 'Subjective Similarity Judgment Task'];
Data_JND_Address=[Data_Address 'Collected Data' filesep 'Near Threshold Discrimination Task'];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

JND_Pairs=[3 4;11 22;6 15;6 14;3 28;6 7;24 30;19 20;4 7;7 20;1 16;13 29;15 19;11 17;14 30;5 30;7 28;15 18;4 20;3 13;9 14;3 9;20 27;2 10];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SubjIDs_Sess={'108169884-1','108169884-2',...
              '695758433-1','695758433-2',...
              '619502508-1','619502508-2',...
              '584740030-1','584740030-2',...
              '089858508-1','089858508-2',...
              '801165888-1','801165888-2',...
              '040062257-1','040062257-2',...
              '940332894-1','940332894-2',...
              '518402380-1','518402380-2',...
              '415090147-1','415090147-2',...
              '951506687-1','951506687-2',...
              '682207766-1','682207766-2',...
              };

JND_Pairs_In_Each_Sess=repmat({[3 4;3 13;4 7;5 30;6 14;7 28;9 14;11 22;14 30;15 19;19 20;20 27], [1 16;2 10;3 9;3 28;4 20;6 7;6 15;7 20;11 17;13 29;15 18;24 30]},1,11);
JND_Pairs_In_Each_Sess{end+1}=[11 22;19 20;24 30;3 28;1 16;4 7;7 28;6 15;13 29;2 10;3 9;5 30];
JND_Pairs_In_Each_Sess{end+1}=[15 18;3 4;6 7;4 20;3 13;14 30;15 19;6 14;7 20;9 14;11 17;20 27];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SubjIDs=SubjIDs_Sess(1:2:end);
Reps=[1 2;3 4;5 6;7 8;9 10;11 12;13 14;15 16;17 18;19 20;21 22;23 24]; %%% Same subject data index %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L1L2=repmat([0.00025 0.00025],24,1); %%% L1,L2 hyperparameters of the ML approach

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
StepsNumberLastToAverage=5;
CorrType='Spearman';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%
PickUpperTriangle=triu(ones(30,30));
PickUpperTriangle(eye(30)==1)=0;
PickUpperTriangle=find(PickUpperTriangle);

%%%%%************************************* Read the data *************************************%%%%%%
Dissim=[];
JNDs=[];
ReversalRatios=[];
Dissim_in_each_sess=[];
WithinSubjsCorrMLVsMDS=[];
for i=1:length(SubjIDs)
    
    %%%%%************ Load Dissim matrices ************%%%%%%
    Dissim_o_in_each_Sess=[];
    Dissim_from_embedding_5d_in_each_sess=[];
    Dissim_ml_in_each_Sess=[];
    disp('************* One Subj ******************')
    disp('## Dissimilarity ##')
    for n=1:length(Reps(i,:))
        %%%% Read the two sessions of the same participants %%%%%
        Files=dir(Data_Sim_Address);
        for j=1:length(Files)
            Check=strsplit(Files(j).name,SubjIDs_Sess{Reps(i,n)});
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

    %%%% Within-subject correlation %%%%
    WithinSubjsCorrMLVsMDS(i,1)=corr(Dissim_ml_in_each_Sess{1}(PickUpperTriangle),Dissim_ml_in_each_Sess{2}(PickUpperTriangle),'Type',CorrType);
    WithinSubjsCorrMLVsMDS(i,2)=corr(Dissim_from_embedding_5d_in_each_sess{1}(PickUpperTriangle),Dissim_from_embedding_5d_in_each_sess{2}(PickUpperTriangle),'Type',CorrType);

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
    Dissim_in_each_sess{Reps(i,1)}=D1;
    Dissim_in_each_sess{Reps(i,2)}=D2;


    %%%%%%************ Load #JNDs ************%%%%%%%%
    disp('## #JNDs ##')
    JNDs{i}=zeros(size(Dissim{i}));
    ReversalRatios{i}=zeros(size(Dissim{i}));
    
    JNDs_In_Each_Sess=[];
    ReversalRatios_In_Each_Sess=[];
    for n=1:length(Reps(i,:))
  
        Files=dir(Data_JND_Address);
        for j=1:length(Files)
            Check=strsplit(Files(j).name,SubjIDs_Sess{Reps(i,n)});
            if(length(Check)>1)
                break;
            end
        end
        Data_JND_Read=[Data_JND_Address filesep Files(j).name filesep];

        disp(Files(j).name)
        JNDs_In_Each_Sess{n}=zeros(size(Dissim{i}));
        ReversalRatios_In_Each_Sess{n}=zeros(size(Dissim{i}));
        for j=1:length(JND_Pairs_In_Each_Sess{Reps(i,n)})
            %%%%% Read each staircase and compute its #JNDs %%%%%%%
            TextFile=fileread([Data_JND_Read 'staircase_' num2str(JND_Pairs_In_Each_Sess{Reps(i,n)}(j,1)) '-' num2str(JND_Pairs_In_Each_Sess{Reps(i,n)}(j,2)) '.log']);
            TextFile=strsplit(TextFile,'responseDistances');
            TextFile=strsplit(TextFile{2},'responseValues');

            DistanceVector=str2num(TextFile{1});
            ChangesPoints=find(diff(DistanceVector)~=0);
            LastToPick=ChangesPoints(end-StepsNumberLastToAverage+1);
            LastToPick=length(DistanceVector)-LastToPick;
            
            JNDs_In_Each_Sess{n}(JND_Pairs_In_Each_Sess{Reps(i,n)}(j,1),JND_Pairs_In_Each_Sess{Reps(i,n)}(j,2))=1000/mean(DistanceVector((end-LastToPick+1):end));
            
            %%%% compute reversal ratios within the last 20 trials %%%%
            TextFile=fileread([Data_JND_Read 'staircase_' num2str(JND_Pairs_In_Each_Sess{Reps(i,n)}(j,1)) '-' num2str(JND_Pairs_In_Each_Sess{Reps(i,n)}(j,2)) '.log']);
            TextFile=strsplit(TextFile,'reversalIndices');
            TextFile=strsplit(TextFile{2},'responseDistances');
            ReversalIndices=str2num(TextFile{1});
            ReversalRatios_In_Each_Sess{n}(JND_Pairs_In_Each_Sess{Reps(i,n)}(j,1),JND_Pairs_In_Each_Sess{Reps(i,n)}(j,2))=sum(ReversalIndices>=40)/20;

        end
    end
    JNDs{i}=JNDs_In_Each_Sess{1}+JNDs_In_Each_Sess{2};
    ReversalRatios{i}=ReversalRatios_In_Each_Sess{1}+ReversalRatios_In_Each_Sess{2};
end
%%%% examined face pairs indexes %%%%
Indexes=sub2ind(size(Dissim{1}),JND_Pairs(:,1),JND_Pairs(:,2));



%%%%%%%%******************************* Compare ML and MDS_5d ************************************%%%%%%%%%
figure('Position', [200 200 900 700]);
[p,h,stats]=signrank(WithinSubjsCorrMLVsMDS(:,1),WithinSubjsCorrMLVsMDS(:,2));
bar(mean(WithinSubjsCorrMLVsMDS),'FaceColor',[0.5 0.5 0.5],'EdgeColor',[0.3 0.3 0.3],'LineWidth',1.5);
hold on;
for i=1:size(WithinSubjsCorrMLVsMDS,1)
    R=0.01*(1-2*rand(1,2));
    plot(1+R(1),WithinSubjsCorrMLVsMDS(i,1),'o','LineWidth',2,'MarkerSize',4,'Color',[0.2 0.2 0.2]);
    plot(2+R(2),WithinSubjsCorrMLVsMDS(i,2),'o','LineWidth',2,'MarkerSize',4,'Color',[0.2 0.2 0.2]);
    plot([1+R(1) 2+R(2)],WithinSubjsCorrMLVsMDS(i,:),'Color',[0 0 0],'LineStyle',':','LineWidth',2);
end
grid on
ylim([0.5 0.9]);
set(gca,'XTick',1:2,'XTickLabel',{'ML','MDS-5d'});
title({'Comparison of ML and MDS approaches', 'in obtaining dissimilarity matrix from subjective similarity ranking data'});
ylabel('Within-subject correlation');
set(gca,'fontsize',14);
print(gcf,'Within-subject correlations ML Vs MDS-5d.png','-dpng','-r300');




%%%%%%*************************** Dissimilarity matrix subject specificity **************************%%%%%%%%
%%%%% Between Vs Within Subjects Correlation %%%%%
Between_Subjects_Corr=[];
k=1;
while(1)
    R=randperm(length(SubjIDs_Sess));
    R=sort(R(1:2));
    if(~ismember(R,Reps,'rows'))
        D1=Dissim_in_each_sess{R(1)}(PickUpperTriangle);
        D2=Dissim_in_each_sess{R(2)}(PickUpperTriangle);
        C=corr(D1,D2,'Type',CorrType);
        Between_Subjects_Corr=[Between_Subjects_Corr; C];
        k=k+1;
    end
    if(k>100000)
        break;
    end
end
Within_Subjects_Corr=[];
Within_Subjects_Corr_z_value=[];
for i=1:length(SubjIDs)
    Within_Subjects_Corr(i)=corr(Dissim_in_each_sess{Reps(i,1)}(PickUpperTriangle),Dissim_in_each_sess{Reps(i,2)}(PickUpperTriangle),'Type',CorrType);
    Within_Subjects_Corr_z_value(i)=(Within_Subjects_Corr(i)-mean(Between_Subjects_Corr))/std(Between_Subjects_Corr);
end

%%%%%%%% Show %%%%%%%%%%%
figure('Position', [0 0 1800 700]);
subplot(1,2,1)
histogram(Between_Subjects_Corr,15,'Normalization','probability');
hold on
Ylim=ylim;
for i=1:length(Within_Subjects_Corr)
    plot([Within_Subjects_Corr(i) Within_Subjects_Corr(i)],Ylim,'--','Color',[1 0 0],'LineWidth',1);
end
grid on;
ylabel('Probability density');
xlabel('Between-participant correlation');
ylim(Ylim);
set(gca,'fontsize',16);

subplot(1,2,2)
bar(mean(Within_Subjects_Corr_z_value),'FaceColor',[0.5 0.5 0.5],'EdgeColor',[0.3 0.3 0.3],'LineWidth',1.5);
%%%%%%%%
NBootstrap=100000;
%%%%%%%%
Bootstrap_Indices=randi([1 length(Within_Subjects_Corr_z_value)], [NBootstrap, length(Within_Subjects_Corr_z_value)]);
Z_mean_distribution=sort(mean(Within_Subjects_Corr_z_value(Bootstrap_Indices),2));
ConfidenceInterval=[Z_mean_distribution(round(0.025*NBootstrap)) Z_mean_distribution(round(0.975*NBootstrap))];
hold on
plot([1, 1],ConfidenceInterval, '-k', 'LineWidth', 5);
set(gca,'XTick',1,'XTickLabel',[]);
ylabel('z-value');
set(gca,'fontsize',16);
grid on;
for i=1:length(Within_Subjects_Corr_z_value)
    plot(1+0.3*(1-2*rand(1)),Within_Subjects_Corr_z_value(i),'o','LineWidth',5,'MarkerSize',8,'Color',[0.2 0.2 0.2]);
end
grid on
print(gcf,['Between- vs within-subject correlation' '_' DissimCalculateApproach '.png'],'-dpng','-r300');



%%%%%%***************************** Show staricase quality check ****************************%%%%%%%%%
figure('Position', [0 0 1920 1080]);
k=1;
for n=Indexes'
    subplot(4,6,k);
    ReversalRatiosAcrossSubjects=[];
    for i=1:length(SubjIDs)
        ReversalRatiosAcrossSubjects(i)=ReversalRatios{i}(n);
    end
    M=mean(ReversalRatiosAcrossSubjects);
    bar(M,'FaceColor',[0.5 0.5 0.5],'EdgeColor',[0.3 0.3 0.3],'LineWidth',1.5);
    hold on
    SD=std(ReversalRatiosAcrossSubjects);
    plot([1, 1],[M-SD, M+SD],'-k', 'LineWidth', 3);
    set(gca,'XTick',1,'XTickLabel', ['Pairs=' num2str(JND_Pairs(k,1)) '-' num2str(JND_Pairs(k,2))]);
    if(mod(k-1,6)==0)
        ylabel({'Reversal Ratio', 'in the last 20 trials'});
    end
    ylim([0 0.5]);
    set(gca,'YTick',0:0.1:0.5);
    set(gca,'fontsize',15);
    grid on;
    for i=1:length(SubjIDs)
        plot(1+0.3*(1-2*rand(1)),ReversalRatiosAcrossSubjects(i),'o','LineWidth',2,'MarkerSize',5,'Color',[0.2 0.2 0.2]);
    end
    k=k+1;
end
print(gcf,'Reversal ratios in each face pair staircase.png','-dpng','-r300');



%%%%%%*************************** correlation between dissimilarity value and #JNDs in each subject ***************************%%%%%%%%
figure('Position', [0 0 1920 1080]);
JND_Dissim_Corr=[];
JND_Dissim_Corr_Pval=[];
JND_Dissim_Corr_Zval=[];
for i=1:length(SubjIDs)
    subplot(3,4,i);
    plot(JNDs{i}(Indexes),Dissim{i}(Indexes),'o','lineWidth',4,'Color',[0 0.6 0.8]);
    grid on;
    [JND_Dissim_Corr(i),JND_Dissim_Corr_Pval(i)]=corr(JNDs{i}(Indexes),Dissim{i}(Indexes),'Type',CorrType);
    %%%%% z-value computation %%%%%
    Fr=0.5*log((1+JND_Dissim_Corr(i))/(1-JND_Dissim_Corr(i)));
    JND_Dissim_Corr_Zval(i)=sqrt((length(Indexes)-3)/1.06)*Fr;

    title(['Subj=' num2str(i) ', r=' num2str(JND_Dissim_Corr(i)) ', p=' num2str(JND_Dissim_Corr_Pval(i))]);
    hold on
    c = polyfit(JNDs{i}(Indexes),Dissim{i}(Indexes),1);
    Xlim=xlim;
    plot((Xlim(1)-1):0.01:(Xlim(2)+1),c(1)*((Xlim(1)-1):0.01:(Xlim(2)+1))+c(2),'--','Color',[0 0.6 0.8],'LineWidth',3)
    xlim(Xlim)
    xlabel('#JNDs');
    ylabel('Dissimilarity value');
    set(gca,'fontsize',16);
    ylim([0 1])
    yticks([0 0.2 0.4 0.6 0.8 1]);
end
print(gcf,'Dissim and #JND correlation.png','-dpng','-r300');



%%%%***************************  correlation between ones dissimilarity value and others' #JNDs ***************************%%%%%%%%
Rvalues=[];
Pvalues=[];
for i=1:length(SubjIDs)
    k=1;
    %%%% Corr with own %%%%
    [Rvalues(i,k),Pvalues(i,k)]=corr(JNDs{i}(Indexes),Dissim{i}(Indexes),'Type',CorrType);
    k=k+1;
    Z_JNDs=0;
    for j=1:length(SubjIDs)
        if(i~=j)
            %%%% Corr with other's %%%%
            [Rvalues(i,k),Pvalues(i,k)]=corr(JNDs{j}(Indexes),Dissim{i}(Indexes),'Type',CorrType);
            k=k+1;

            JNDsHere=JNDs{j}(Indexes);
            Z_JNDs=Z_JNDs+(JNDsHere-mean(JNDsHere))/std(JNDsHere);
        end
    end
    %%%% Corr with group average %%%%
    [Rvalues(i,k),Pvalues(i,k)]=corr(Z_JNDs,Dissim{i}(Indexes),'Type',CorrType);
end



%%%%%%%%%%******************************* Permutation Test *******************************%%%%%%%%%%%
%%%%%******%%%%%%%
NPermutations=100000;
PThresh=0.05; %%%% Individual-level significance threshold %%%%%%
%%%%%*****%%%%%
%%%%%% z-score %%%%%%
ALLSubjectsJNDsZscored=[];
for i=1:length(SubjIDs)
    JNDsHere=JNDs{i}(Indexes);
    ALLSubjectsJNDsZscored(:,i)=(JNDsHere-mean(JNDsHere))/std(JNDsHere);
end
CorrOwn=[];
CorrPerm=[];
SigCorrPermutationTestLine=[];
PvaluesPermutationTest=[];
ZvaluesPermutationTest=[];
for i=1:length(SubjIDs)
    DissimilaritySubject=Dissim{i}(Indexes);
    CorrOwn(i)=corr(ALLSubjectsJNDsZscored(:,i),DissimilaritySubject,'Type',CorrType);
    %%%% Compute null distributions --> CorrPerm %%%%
    for j=1:NPermutations
        listToChooseFrom=1:size(ALLSubjectsJNDsZscored,2);
        listToChooseFrom(listToChooseFrom==i)=[];
        Rinitial=randi([1 length(listToChooseFrom)],size(ALLSubjectsJNDsZscored,1),1);
        R=listToChooseFrom(Rinitial)';

        IndexesSelectHere=sub2ind(size(ALLSubjectsJNDsZscored),(1:size(ALLSubjectsJNDsZscored,1))',R);
        PermutatedJND=ALLSubjectsJNDsZscored(IndexesSelectHere);
        CorrPerm(i,j)=corr(PermutatedJND,DissimilaritySubject,'Type',CorrType);
    end
    %%%%% Evaluate significance level %%%%%
    Limit=round(PThresh*NPermutations);
    CorrPermSorted=fliplr(sort(CorrPerm(i,:)));
    SigCorrPermutationTestLine(i)=CorrPermSorted(Limit);
    if(isempty(find(CorrOwn(i)>=CorrPermSorted,1)))
        PvaluesPermutationTest(i)=1;
    else
        PvaluesPermutationTest(i)=find(CorrOwn(i)>=CorrPermSorted,1)/NPermutations;
    end
    ZvaluesPermutationTest(i)=(CorrOwn(i)-mean(CorrPermSorted))/std(CorrPermSorted);
end

%%%%%% Visualization %%%%%%%%%%%
Color_List=[0 0 1;1 0 0;0 1 0];
figure('Position', [0 0 1300 1080]);
hold on
for i=1:size(Rvalues,1)
    for j=1:size(Rvalues,2)
        switch j
            case 1
                %%%% Corr with own #JNDs %%%%%
                ColorChoose=Color_List(1,:);
                RLoc=i+0.3*rand(1)-0.15;
                plot(RLoc,Rvalues(i,j),'s','LineWidth',13,'Color',ColorChoose);

            case size(Rvalues,1)+1
                %%%% Corr with group-averaged #JNDs %%%%%
                ColorChoose=Color_List(3,:);
                plot(i+0.2*rand(1)-0.1,Rvalues(i,j),'d','LineWidth',8,'Color',ColorChoose);
                
            otherwise
                %%%% Corr with others #JNDs %%%%%
                ColorChoose=Color_List(2,:);
                plot(i+0.3*rand(1)-0.15,Rvalues(i,j),'o','LineWidth',6,'Color',ColorChoose);
        end
        
    end
    plot([i-0.5 i+0.5],[SigCorrPermutationTestLine(i) SigCorrPermutationTestLine(i)],'LineStyle','--','LineWidth',2,'Color',[0 0 0]);
end
xticks(1:size(Rvalues,1))
xlim([0.5 size(Rvalues,1)+0.5])
ylabel('Correlation (r)');
grid on
set(gca,'fontsize',20);
Ylim=ylim;
for i=1:size(Rvalues,1)
    text(i-0.125,Ylim(2)+Ylim(2)/20,['p:' num2str(PvaluesPermutationTest(i),'%0.5f')],'Color',[0 0 1],'FontSize',10);
end
print(gcf,'Individual specificity of the correlations between dissim  and #JNDs','-dpng','-r300');



%%%%%********************************************* Group-level statistics ***********************************************%%%%%
%%%%%%%%%%%% Hypothesis 1 %%%%%%%%%%%%
%%% Bootstraping to find the confidence interval %%%%%%
%%%%%%%%
NBootstrap=100000;
%%%%%%%%
Bootstrap_Indices=randi([1 length(JND_Dissim_Corr_Zval)], [NBootstrap, length(JND_Dissim_Corr_Zval)]);
Z_mean_distribution=sort(mean(JND_Dissim_Corr_Zval(Bootstrap_Indices),2));
ConfidenceInterval1=[Z_mean_distribution(round(0.025*NBootstrap)) Z_mean_distribution(round(0.975*NBootstrap))];
%%% T-test: BF, p-value %%%
[BF1_0,p_value1_0]=bf.ttest(JND_Dissim_Corr_Zval,0*ones(1,length(JND_Dissim_Corr_Zval)));
%%% Fisher's test %%%
X2=-2*sum(log(JND_Dissim_Corr_Pval));
FishersP1=1-chi2cdf(X2,2*length(JND_Dissim_Corr_Pval));
%%%%%%%%%%%% Hypothesis 2 %%%%%%%%%%%%
%%% Bootstraping to find the confidence interval %%%%%%
%%%%%%%%
NBootstrap=100000;
%%%%%%%%
Bootstrap_Indices=randi([1 length(ZvaluesPermutationTest)], [NBootstrap, length(ZvaluesPermutationTest)]);
Z_mean_distribution2=sort(mean(ZvaluesPermutationTest(Bootstrap_Indices),2));
ConfidenceInterval2=[Z_mean_distribution2(round(0.025*NBootstrap)) Z_mean_distribution2(round(0.975*NBootstrap))];
%%% T-test: BF, p-value %%%
[BF2_0,p_value2_0]=bf.ttest(ZvaluesPermutationTest,0*ones(1,length(ZvaluesPermutationTest)));
%%% Fisher's test %%%
X2=-2*sum(log(PvaluesPermutationTest));
FishersP2=1-chi2cdf(X2,2*length(PvaluesPermutationTest));

%%%%%%%%%%%%%%%% Show the stats %%%%%%%%%%%%%%%%%%%%%%%%
figure('Position', [0 0 1200 600])
%%%% Hypothesis 1 %%%%%%
subplot(1,2,1);
bar(mean(JND_Dissim_Corr_Zval),'FaceColor',[0.5 0.5 0.5],'EdgeColor',[0.3 0.3 0.3],'LineWidth',1.5);
hold on
plot([1, 1],ConfidenceInterval1, '-k', 'LineWidth', 5);
set(gca,'XTick',1,'XTickLabel','Hypothesis I');
ylabel('z-value');
set(gca,'fontsize',23);
grid on;
for i=1:length(JND_Dissim_Corr_Zval)
    plot(1+0.3*(1-2*rand(1)),JND_Dissim_Corr_Zval(i),'o','LineWidth',5,'MarkerSize',8,'Color',[0.2 0.2 0.2]);
end
Ylim=ylim;
Xlim=xlim;
text(Xlim(1)+0.05,Ylim(2)-0.1,['95% CI:[' num2str(ConfidenceInterval1(1)) ', ' num2str(ConfidenceInterval1(2)) ']'],'FontSize',12);
text(Xlim(1)+0.05,Ylim(2)-0.3,['t-test ' '; BF:' num2str(BF1_0) ', Pval:' num2str(p_value1_0)],'FontSize',12);
text(Xlim(1)+0.05,Ylim(2)-0.5,['Fisher''s p-value: ' num2str(FishersP1)],'FontSize',12);
%%%% Hypothesis 2 %%%%%%
subplot(1,2,2);
bar(mean(ZvaluesPermutationTest),'FaceColor',[0.5 0.5 0.5],'EdgeColor',[0.3 0.3 0.3],'LineWidth',1.5);
hold on
plot([1, 1],ConfidenceInterval2, '-k', 'LineWidth', 5);
set(gca,'XTick',1,'XTickLabel','Hypothesis II');
ylabel('z-value');
set(gca,'fontsize',23);
grid on;
for i=1:length(ZvaluesPermutationTest)
    plot(1+0.3*(1-2*rand(1)),ZvaluesPermutationTest(i),'o','LineWidth',5,'MarkerSize',8,'Color',[0.2 0.2 0.2]);
end
Ylim=ylim;
Xlim=xlim;
text(Xlim(1)+0.05,Ylim(2)-0.1,['95% CI:[' num2str(ConfidenceInterval2(1)) ', ' num2str(ConfidenceInterval2(2)) ']'],'FontSize',12);
text(Xlim(1)+0.05,Ylim(2)-0.3,['t-test ' '; BF:' num2str(BF2_0) ', Pval:' num2str(p_value2_0)],'FontSize',12);
text(Xlim(1)+0.05,Ylim(2)-0.5,['Fisher''s p-value: ' num2str(FishersP2)],'FontSize',12);
print(gcf,'Group-level statistics.png','-dpng','-r300');



%%%%%%%***************** relationship between dissimilarity value and #JNDs in each face pair across subjects *******************%%%%%%%%%%%
%%%%% z_score %%%%%
z_JNDs=[];
z_Dissim=[];
for i=1:length(SubjIDs)
    z_JNDs{i}=zeros(size(JNDs{i}));
    z_JNDs{i}(Indexes)=(JNDs{i}(Indexes)-mean(JNDs{i}(Indexes)))/std(JNDs{i}(Indexes));
    z_Dissim{i}=(Dissim{i}-mean(Dissim{i}(PickUpperTriangle)))/std(Dissim{i}(PickUpperTriangle));
end

z_JNDPairs=[];
z_DissimPairs=[];
MedianDissimPairs=[];
MadDissimPairs=[];
for i=1:size(JND_Pairs,1)
    for j=1:length(SubjIDs)
        z_JNDPairs{i}(j)=z_JNDs{j}(JND_Pairs(i,1),JND_Pairs(i,2));
        z_DissimPairs{i}(j)=z_Dissim{j}(JND_Pairs(i,1),JND_Pairs(i,2));
    end
    MedianDissimPairs(i)=median(z_DissimPairs{i});
    MadDissimPairs(i)=mad(z_DissimPairs{i},1); %%% Controversy level
end

figure('Position', [0 0 1920 1080]);
CorrPairs=[];
PValPairs=[];
[~,ShowOrder]=sort(MadDissimPairs);
ShowOrder=fliplr(ShowOrder); 
k=1;
for i=ShowOrder
    subplot(4,6,k);

    plot(z_JNDPairs{i},z_DissimPairs{i},'o','MarkerSize',5,'lineWidth',3,'Color',[0 0.6 0.8]);
    grid on;
    hold on;
    Ylim=ylim;
    ylim([Ylim(1)-0.2 Ylim(2)+0.2]);
    c = polyfit(z_JNDPairs{i}',z_DissimPairs{i}',1);
    Xlim=xlim;
    plot((Xlim(1)-1):0.01:(Xlim(2)+1),c(1)*((Xlim(1)-1):0.01:(Xlim(2)+1))+c(2),'--','Color',[0 0.6 0.8],'LineWidth',2.5)
    xlim([Xlim(1)-0.2 Xlim(2)+0.2]);
    [CorrPairs(i),PValPairs(i)]=corr(z_JNDPairs{i}',z_DissimPairs{i}','Type',CorrType);
    xlabel('#JNDs(z-normalized)');
    ylabel('Dissimilarity value(z-normalized)');
    
    set(gca,'fontsize',9);
    k=k+1;
end

k=1;
for i=ShowOrder
    subplot(4,6,k);
    title({['Pairs=' num2str(JND_Pairs(i,1)) '-' num2str(JND_Pairs(i,2))]},{['r=' num2str(CorrPairs(i),'%.2f') ', p=' num2str(PValPairs(i),'%.4f')]});
    k=k+1;
end
print(gcf,'Correlation between dissim and #JNDs in each face pair.png','-dpng','-r300');
[BF3,p_value3]=bf.ttest(CorrPairs,0*ones(1,length(CorrPairs)));

close all;

%%%%%******************** Bayesian Prevalence ************************%%%%%%
addpath([pwd filesep 'bayesian-prevalence-master\matlab']);
alpha=0.05; % this specifies the alpha value used for the within-unit tests
Ntests=12; % number of participants
Nsigtests=5;% number of significant at the individual-level

x=linspace(0,1,100);
posterior=bayesprev_posterior(x,Nsigtests,Ntests,alpha);
xmap=bayesprev_map(Nsigtests,Ntests,alpha);
pmap=bayesprev_posterior(xmap,Nsigtests,Ntests,alpha);
%figure
%plot(x,posterior,'LineWidth',4);
h=bayesprev_hpdi(0.95,Nsigtests,Ntests,alpha,1);
%hold on
%plot([h(1) h(2)],[pmap pmap],'LineWidth',2)

