clc;
clear;

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
JND_Pairs=[3 5;13 20;4 12;14 27;15 24;14 16;5 17;10 17;7 12;1 28;13 30;22 23;2 10];
SubjIDs={'960425785','674231075','664024137','596926644'};
%%%%
StepsNumberLastToAverage=5;
CorrType='Spearman';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%% Read the data %%%%%%%%%%%%%%%%%%%%%%
Dissim=[];
JNDs=[];
for i=1:length(SubjIDs)
    %%%%% Load the dissim matrix %%%%%%
    Files=dir(Data_Sim_Address);
    for j=1:length(Files)
        Check=strsplit(Files(j).name,SubjIDs{i});
        if(length(Check)>1)
            break;
        end
    end
    load([Data_Sim_Address filesep Files(j).name filesep 'dissim_matrix_from_embedding.csv'])
    Dissim{i}=dissim_matrix_from_embedding;

    %%%%% Load JNDs %%%%%%
    Files=dir(Data_JND_Address);
    for j=1:length(Files)
        Check=strsplit(Files(j).name,SubjIDs{i});
        if(length(Check)>1)
            break;
        end
    end
    Data_JND_Read=[Data_JND_Address filesep Files(j).name filesep];
    
    JNDs{i}=zeros(size(Dissim{i}));
    for j=1:length(JND_Pairs)
        TextFile=fileread([Data_JND_Read 'staircase_' num2str(JND_Pairs(j,1)) '-' num2str(JND_Pairs(j,2)) '.log']);
        TextFile=strsplit(TextFile,'responseDistances');
        TextFile=strsplit(TextFile{2},'responseValues');
    
        DistanceVector=str2num(TextFile{1});
        ChangesPoints=find(diff(DistanceVector)~=0);
        LastToPick=ChangesPoints(end-StepsNumberLastToAverage+1);
        LastToPick=length(DistanceVector)-LastToPick;

        JNDs{i}(JND_Pairs(j,1),JND_Pairs(j,2))=1000/mean(DistanceVector((end-LastToPick+1):end));
    end
end
Indexes=sub2ind(size(Dissim{1}),JND_Pairs(:,1),JND_Pairs(:,2));

%%%%%%%%%%%%%%%%%%%%%%% Within Subject correlation %%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Position', [0 0 1080 1080]);
WithinCorr=[];
WithinCorrPval=[];
WithinCorrZval=[];
for i=1:length(SubjIDs)
    subplot(2,2,i);
    plot(JNDs{i}(Indexes),Dissim{i}(Indexes),'o','lineWidth',4,'Color',[0 0.6 0.8]);
    grid on;
    [WithinCorr(i),WithinCorrPval(i)]=corr(JNDs{i}(Indexes),Dissim{i}(Indexes),'Type',CorrType);
    %%%%% Z-value estimation %%%%%
    Fr=0.5*log((1+WithinCorr(i))/(1-WithinCorr(i)));
    WithinCorrZval(i)=sqrt((length(Indexes)-3)/1.06)*Fr;

    title(['Subj=' num2str(i) ', r=' num2str(WithinCorr(i)) ', p=' num2str(WithinCorrPval(i))]);
    hold on
    c = polyfit(JNDs{i}(Indexes),Dissim{i}(Indexes),1);
    Xlim=xlim;
    plot((Xlim(1)-1):0.01:(Xlim(2)+1),c(1)*((Xlim(1)-1):0.01:(Xlim(2)+1))+c(2),'--','Color',[0 0.6 0.8],'LineWidth',3)
    xlim(Xlim)
    xlabel('#JNDs');
    ylabel('Dissimilarity value');
    set(gca,'fontsize',16);
    ylim([0.1 1])
end
print(gcf,'Within-Subject-Correlations.png','-dpng','-r300');

%%%%%%%%%%%%%%%%%%%%%%%%% Between Subjects %%%%%%%%%%%%%%%%%%%%%%%%%%%%
Rvalues=[];
Pvalues=[];
for i=1:length(SubjIDs)
    k=1;
    [Rvalues(i,k),Pvalues(i,k)]=corr(JNDs{i}(Indexes),Dissim{i}(Indexes),'Type',CorrType);
    k=k+1;
    Z_JNDs=0;
    for j=1:length(SubjIDs)
        if(i~=j)
            [Rvalues(i,k),Pvalues(i,k)]=corr(JNDs{j}(Indexes),Dissim{i}(Indexes),'Type',CorrType);
            k=k+1;
        end
        JNDsHere=JNDs{j}(Indexes);
        Z_JNDs=Z_JNDs+(JNDsHere-mean(JNDsHere))/std(JNDsHere);

    end
    [Rvalues(i,k),Pvalues(i,k)]=corr(Z_JNDs,Dissim{i}(Indexes),'Type',CorrType);
end

%%%%%% Permutation Test %%%%%%%
%%%%%******%%%%%%%
NPermutations=100000;
PThresh=0.05;
%%%%%*****%%%%%
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
    for j=1:NPermutations
        listToChooseFrom=1:size(ALLSubjectsJNDsZscored,2);
        listToChooseFrom(listToChooseFrom==i)=[];
        Rinitial=randi([1 length(listToChooseFrom)],size(ALLSubjectsJNDsZscored,1),1);
        R=listToChooseFrom(Rinitial)';

        IndexesSelectHere=sub2ind(size(ALLSubjectsJNDsZscored),(1:size(ALLSubjectsJNDsZscored,1))',R);
        PermutatedJND=ALLSubjectsJNDsZscored(IndexesSelectHere);
        CorrPerm(i,j)=corr(PermutatedJND,DissimilaritySubject,'Type',CorrType);
    end
    Limit=round(PThresh*NPermutations);
    CorrPermSorted=fliplr(sort(CorrPerm(i,:)));
    SigCorrPermutationTestLine(i)=CorrPermSorted(Limit);
    PvaluesPermutationTest(i)=find(CorrOwn(i)>=CorrPermSorted,1)/NPermutations;
    ZvaluesPermutationTest(i)=(CorrOwn(i)-mean(CorrPermSorted))/std(CorrPermSorted);
end

%%%%%% Visualization %%%%%%%%%%%
Color_List=[0 0 1;1 0 0;0 1 0];
figure('Position', [0 0 1080 1080]);
hold on
for i=1:size(Rvalues,1)
    for j=1:size(Rvalues,2)
        switch j
            case 1
                ColorChoose=Color_List(1,:);
                RLoc=i+0.3*rand(1)-0.15;
                plot(RLoc,Rvalues(i,j),'s','LineWidth',18,'Color',ColorChoose);

            case size(Rvalues,1)+1
                ColorChoose=Color_List(3,:);
                plot(i+0.2*rand(1)-0.1,Rvalues(i,j),'d','LineWidth',15,'Color',ColorChoose);
                
            otherwise
                ColorChoose=Color_List(2,:);
                plot(i+0.3*rand(1)-0.15,Rvalues(i,j),'o','LineWidth',12,'Color',ColorChoose);
        end
        
    end
    plot([i-0.5 i+0.5],[SigCorrPermutationTestLine(i) SigCorrPermutationTestLine(i)],'LineStyle','--','LineWidth',2,'Color',[0 0 0]);
end
xticks(1:size(Rvalues,1))
xlim([0.5 size(Rvalues,1)+0.5])
ylabel('Correlation (r)');
grid on
set(gca,'fontsize',16);
Ylim=ylim;
for i=1:size(Rvalues,1)
    text(i-0.125,Ylim(2)+Ylim(2)/20,['p:' num2str(PvaluesPermutationTest(i),'%0.5f')],'Color',[0 0 1],'FontSize',18);
end
print(gcf,'Between-Subjects_Correlations','-dpng','-r300');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Group-Stats %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% Hypothesis 1 %%%%%%%%%%%%
%%%Bootstraping to find the confidence interval%%%%%%
%%%%%%%%
NBootstrap=100000;
%%%%%%%%
Bootstrap_Indices=randi([1 length(WithinCorrZval)], [NBootstrap, length(WithinCorrZval)]);
Z_mean_distribution=sort(mean(WithinCorrZval(Bootstrap_Indices),2));
ConfidenceInterval1=[Z_mean_distribution(round(0.05*NBootstrap)) Z_mean_distribution(round(0.95*NBootstrap))];
%%% T-test: BF, p-value %%%
[BF1_0,p_value1_0]=bf.ttest(WithinCorrZval,0*ones(1,length(WithinCorrZval)));
%%% Fisher's test %%%
X2=-2*sum(log(WithinCorrPval));
FishersP1=1-chi2cdf(X2,2*length(WithinCorrPval));
%%%%%%%%%%%% Hypothesis 2 %%%%%%%%%%%%
%%%Bootstraping to find the confidence interval%%%%%%
%%%%%%%%
NBootstrap=100000;
%%%%%%%%
Bootstrap_Indices=randi([1 length(ZvaluesPermutationTest)], [NBootstrap, length(ZvaluesPermutationTest)]);
Z_mean_distribution2=sort(mean(ZvaluesPermutationTest(Bootstrap_Indices),2));
ConfidenceInterval2=[Z_mean_distribution2(round(0.05*NBootstrap)) Z_mean_distribution2(round(0.95*NBootstrap))];
%%% T-test: BF, p-value %%%
[BF2_0,p_value2_0]=bf.ttest(ZvaluesPermutationTest,0*ones(1,length(ZvaluesPermutationTest)));
%%% Fisher's test %%%
X2=-2*sum(log(PvaluesPermutationTest));
FishersP2=1-chi2cdf(X2,2*length(PvaluesPermutationTest));

%%%%%%%%%%%%%%%% Show the stats %%%%%%%%%%%%%%%%%%%%%%%%
figure('Position', [0 0 1200 600])
%%%% Hypothesis 1 %%%%%%
subplot(1,2,1);
bar(mean(WithinCorrZval),'FaceColor',[0.5 0.5 0.5],'EdgeColor',[0.3 0.3 0.3],'LineWidth',1.5);
hold on
plot([1, 1],ConfidenceInterval1, '-k', 'LineWidth', 5);
set(gca,'XTick',1,'XTickLabel','Hypothesis I');
ylabel('z-value');
set(gca,'fontsize',23);
grid on;
for i=1:length(WithinCorrZval)
    plot(1+0.3*(1-2*rand(1)),WithinCorrZval(i),'o','LineWidth',5,'MarkerSize',8,'Color',[0.2 0.2 0.2]);
end
Ylim=ylim;
Xlim=xlim;
text(Xlim(1)+0.05,Ylim(2)-0.1,['95% CI:[' num2str(ConfidenceInterval1(1)) ', ' num2str(ConfidenceInterval1(2)) ']'],'FontSize',15);
text(Xlim(1)+0.05,Ylim(2)-0.25,['t-test ' '; BF:' num2str(BF1_0) ', Pval:' num2str(p_value1_0)],'FontSize',15);
text(Xlim(1)+0.05,Ylim(2)-0.4,['Fisher''s p-value: ' num2str(FishersP1)],'FontSize',15);
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
text(Xlim(1)+0.05,Ylim(2)-0.1,['95% CI:[' num2str(ConfidenceInterval2(1)) ', ' num2str(ConfidenceInterval2(2)) ']'],'FontSize',15);
text(Xlim(1)+0.05,Ylim(2)-0.25,['t-test ' '; BF:' num2str(BF2_0) ', Pval:' num2str(p_value2_0)],'FontSize',15);
text(Xlim(1)+0.05,Ylim(2)-0.4,['Fisher''s p-value: ' num2str(FishersP2)],'FontSize',15);
print(gcf,'Group_Statistics.png','-dpng','-r300');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PairWise %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
z_JNDs=[];
z_Dissim=[];
for i=1:length(SubjIDs)
    z_JNDs{i}=zeros(size(JNDs{i}));
    z_JNDs{i}(Indexes)=(JNDs{i}(Indexes)-mean(JNDs{i}(Indexes)))/std(JNDs{i}(Indexes));
    z_Dissim{i}=(Dissim{i}-mean(Dissim{i}(:)))/std(Dissim{i}(:));
end

figure('Position', [0 0 1920 1080]);
z_JNDofPairs=[];
z_DissimofPairs=[];
CorrPairs=[];
PvalPairs=[];
AvgDissimPairs=[];
stdDissimPairs=[];
for i=1:size(JND_Pairs,1)
    subplot(3,5,i);
    for j=1:length(SubjIDs)
        z_JNDofPairs{i}(j)=z_JNDs{j}(JND_Pairs(i,1),JND_Pairs(i,2));
        z_DissimofPairs{i}(j)=z_Dissim{j}(JND_Pairs(i,1),JND_Pairs(i,2));
    end
    plot(z_JNDofPairs{i},z_DissimofPairs{i},'o','lineWidth',4,'Color',[0 0.6 0.8]);
    grid on;
    hold on;
    Ylim=ylim;
    ylim([Ylim(1)-0.2 Ylim(2)+0.2]);
    c = polyfit(z_JNDofPairs{i}',z_DissimofPairs{i}',1);
    Xlim=xlim;
    plot((Xlim(1)-1):0.01:(Xlim(2)+1),c(1)*((Xlim(1)-1):0.01:(Xlim(2)+1))+c(2),'--','Color',[0 0.6 0.8],'LineWidth',3)
    xlim([Xlim(1)-0.2 Xlim(2)+0.2]);
    [CorrPairs(i),PValPairs(i)]=corr(z_JNDofPairs{i}',z_DissimofPairs{i}','Type',CorrType);
    title(['Pairs=' num2str(JND_Pairs(i,1)) '-' num2str(JND_Pairs(i,2)) ', r=' num2str(CorrPairs(i))]);
    xlabel('#JNDs(z-normalized)');
    ylabel('Dissimilarity value(z-normalized)');
    
    AvgDissimPairs(i)=mean(z_DissimofPairs{i});
    stdDissimPairs(i)=std(z_DissimofPairs{i});
    set(gca,'fontsize',11);
end
print(gcf,'PairWiseCorrelations.png','-dpng','-r300');


figure('Position', [0 0 600 600]);
plot(stdDissimPairs,CorrPairs,'+','LineWidth',16,'Color',[0 0.6 0.8])
r=corr(stdDissimPairs',CorrPairs');
hold on
c = polyfit(stdDissimPairs',CorrPairs',1);
Xlim=xlim;
plot((Xlim(1)-1):0.01:(Xlim(2)+1),c(1)*((Xlim(1)-1):0.01:(Xlim(2)+1))+c(2),'--','Color',[0 0.6 0.8],'LineWidth',3)
xlim(Xlim)
ylabel('#JNDs and dissimilairty correlation in face pairs');
xlabel('std of the dissimilarity value across participants')
title(['r=' num2str(r)]);
grid on
set(gca,'fontsize',16);
ylim([-1 1])
print(gcf,'DissimilarityControversyVSCorrelation.png','-dpng','-r300');

close all;
