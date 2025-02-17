clc;
clear;
rng("default")

%%%%%%%%%%**************** Choose dissimilarity matrix calculation approach **************%%%%%%%%%%%
DissimCalculateApproach='ML'; %%% 'ML','MDS_5d','MDS_2d'
EnablePlot=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SubjIDs={'108169884-1','108169884-2',...
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


%%%%% Load the dissim matrix %%%%%%
Dissim=[];
SessionQuality=[];
for n=1:length(SubjIDs)
    
    Files=dir(Data_Sim_Address);
    for j=1:length(Files)
        Check=strsplit(Files(j).name,SubjIDs{n});
        if(length(Check)>1)
            break;
        end
    end

    disp('********************');
    disp(Files(j).name);
    
    switch DissimCalculateApproach

        case 'MDS_5d'
            load([Data_Sim_Address filesep Files(j).name filesep 'dissim_matrix.csv']);
            dissim_matrix(dissim_matrix==0)=nan;
            dissim_matrix(eye(size(dissim_matrix))==1)=0;
            Dissim{n}=dist(mdscale(dissim_matrix,5,'Criterion','metricstress','Start','random')')';
            Dissim{n}=Dissim{n}/max(Dissim{n}(:));

        case 'MDS_2d'
            load([Data_Sim_Address filesep Files(j).name filesep 'dissim_matrix.csv']);
            dissim_matrix(dissim_matrix==0)=nan;
            dissim_matrix(eye(size(dissim_matrix))==1)=0;
            Dissim{n}=dist(mdscale(dissim_matrix,2,'Criterion','metricstress','Start','random')')';
            Dissim{n}=Dissim{n}/max(Dissim{n}(:));

        case 'ML'
            ML_Approach_Embeddings=readtable([Data_Sim_Address filesep Files(j).name filesep 'ML_Approach' filesep 'Space_' num2str(L1L2(n,1)) '_' num2str(L1L2(n,2)) '.csv']);
            ML_Approach_Embeddings=table2array(ML_Approach_Embeddings(2:end,:));
            Std_Embeddings=std(ML_Approach_Embeddings); 
            Std_Embeddings_Percent=100*Std_Embeddings/sum(Std_Embeddings);
            Dissim{n}=dist(ML_Approach_Embeddings');
            Dissim{n}=Dissim{n}/max(Dissim{n}(:));
    end

    %%%%%%%%%%% QualityCheck within a session (Test-retest, correlation between the first and second-half) %%%%%%%%%%%
    ClickData=readtable([Data_Sim_Address filesep Files(j).name filesep 'click_data.csv']);
    AllClicks=length(table2array(ClickData(:,1)));
    %%% First half DissimMatrix %%%
    DissimMatrixFirstHalf=MakeDissimFromClickLoggings(ClickData,1:round(AllClicks/2),length(Dissim{n}));
    DissimMatrixFirstHalf=dist(mdscale(DissimMatrixFirstHalf,5,'Criterion','metricstress','Start','random')');
    %%% Second half DissimMatrix %%%
    DissimMatrixSecondHalf=MakeDissimFromClickLoggings(ClickData,(round(AllClicks/2)+1):AllClicks,length(Dissim{n}));
    DissimMatrixSecondHalf=dist(mdscale(DissimMatrixSecondHalf,5,'Criterion','metricstress','Start','random')');
    %%%% Correlation between first and second half %%%
    PickUpperTriangle=triu(ones(length(DissimMatrixFirstHalf),length(DissimMatrixFirstHalf)));
    PickUpperTriangle(eye(length(DissimMatrixFirstHalf))==1)=0;
    PickUpperTriangle=find(PickUpperTriangle);
    FirstHalf=DissimMatrixFirstHalf(PickUpperTriangle);
    SecondHalf=DissimMatrixSecondHalf(PickUpperTriangle);
    SessionQuality(n)=corr(FirstHalf,SecondHalf,'Type','Spearman');
    
    disp(['Subject: ' SubjIDs{n}]);
    disp(['first-second half corr:' num2str(SessionQuality(n))]);

    if(EnablePlot)
        figure
        imagesc(Dissim{n},[0 1]);
        colormap(gray)
        axis off
        colorbar
        set(gca,'fontsize',16);
        print(gcf,['Visualization Dissimilarity Matrix-' SubjIDs{n} '_' DissimCalculateApproach '.png'],'-dpng','-r300');

        switch DissimCalculateApproach
            case {'MDS_2d','MDS_5d'}
                dissim_matrix(dissim_matrix==0)=nan;
                dissim_matrix(eye(size(dissim_matrix))==1)=0;
                D=mdscale(dissim_matrix,2,'Criterion','metricstress','Start','random');

            case {'ML'}
                D=mdscale(Dissim{n},2,'Criterion','metricstress','Start','random');
        end

        D=Rotate_Same_Ref(D);
        figure('Position', [0 0 1920 1080]);
        for i=1:length(D)
            plot(D(i,1),D(i,2),'o');
            hold on
        end
        ax=gca;
        xscale=get(ax,'xlim');
        yscale=get(ax,'ylim');
        xlim([-max(abs(xscale)) max(abs(xscale))]);
        ylim([-max(abs(yscale)) max(abs(yscale))]);
        grid on;
        set(gca,'fontsize',20);
        ax=gca;
        AxesPos=get(ax,'position');
        xscale=get(ax,'xlim');
        yscale=get(ax,'ylim');

        ImgSize=0.11;
        for i=1:length(D)
            Xnorm=((D(i,1)-xscale(1))/diff(xscale))*AxesPos(3)+AxesPos(1)-ImgSize/2;
            Ynorm=((D(i,2)-yscale(1))/diff(yscale))*AxesPos(4)+AxesPos(2)-ImgSize/2;
            ax2=axes('position',[Xnorm Ynorm ImgSize ImgSize]);
            box on;
            if(i>=10)
                Image=imread(['faces' filesep 'Face' num2str(i) '.png']);
            else
                Image=imread(['faces' filesep 'Face0' num2str(i) '.png']);
            end
            imshow(Image(100:1330,481:1600,:));
            axis off;
        end
        grid on
        print(gcf,['2D Visualization Dissimilarity Matrix-' SubjIDs{n} '_' DissimCalculateApproach '.png'],'-dpng','-r300');
        close all
    end
end



function DissimMatrix=MakeDissimFromClickLoggings(ClickData,clicksListIndex,NMatrix)
    
    SimTracker=[];
    for i=1:NMatrix
        SimTracker{i}=[];
    end
    
    %%%%% Read Targets and Clicks %%%%%
    Trials=table2array(ClickData(:,2));
    Clicks=table2array(ClickData(:,5))+1;
    Targets=table2array(ClickData(:,8))+1;


    %%%%% Save trio segments in SimTracker %%%%%
    ValdiTrialflag=0;
    TrialStart=Trials(clicksListIndex(1));
    for i=clicksListIndex
        if(Trials(i)==TrialStart)
            ValdiTrialflag=ValdiTrialflag+1;
        else
            TrialStart=Trials(i);
            ValdiTrialflag=1;
        end
        if(ValdiTrialflag==4)
            for n=0:2
                for m=(n+1):3
                    SimTracker{Targets(i)}=[SimTracker{Targets(i)};Clicks(i-3+n) Clicks(i-3+m)];
                end
            end
        end
    end
    
    %%%%% Derive dissim matrix from SimTracker %%%%%
    DissimMatrix=NaN*ones(NMatrix,NMatrix);
    DissimMatrix(eye(NMatrix)==1)=0;
    for i=1:NMatrix
        for j=1:NMatrix 
            All=sum(sum(ismember(SimTracker{i},j),2)==1)+sum(sum(ismember(SimTracker{j},i),2)==1);
            NonOdds=sum(ismember(SimTracker{i}(:,1),j))+sum(ismember(SimTracker{j}(:,1),i));
            if(All~=0)
                if(NonOdds==0)
                    DissimMatrix(i,j)=1-0.5/(All+0.5);
                else
                    DissimMatrix(i,j)=1-NonOdds/(All+0.5);
                end
            end
        end
    end
end
