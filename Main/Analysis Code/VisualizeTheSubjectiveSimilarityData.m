clc;
clear;

SubjIDs=%% No data collected yet %%

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
for n=1:length(SubjIDs)
    
    Files=dir(Data_Sim_Address);
    for j=1:length(Files)
        Check=strsplit(Files(j).name,SubjIDs{n});
        if(length(Check)>1)
            break;
        end
    end
    load([Data_Sim_Address filesep Files(j).name filesep 'dissim_matrix_from_embedding.csv'])
    Dissim{n}=dissim_matrix_from_embedding;

    %%%%%%%%%%% QualityCheck (Test-retest, correlation between the first and second-half) %%%%%%%%%%%
    ClickData=readtable([Data_Sim_Address filesep Files(j).name filesep 'click_data.csv']);
    AllClicks=length(table2array(ClickData(:,1)));
    %%% First half DissimMatrix %%% 
    DissimMatrixFirstHalf=MakeDissimFromClickLoggings(ClickData,1:round(AllClicks/2),length(Dissim{n}));
    DissimMatrixFirstHalf=dist(mdscale(DissimMatrixFirstHalf,5,'Criterion','metricstress','Start','random')');
    %%% Second half DissimMatrix %%% 
    DissimMatrixSecondHalf=MakeDissimFromClickLoggings(ClickData,round(AllClicks/2):AllClicks,length(Dissim{n}));
    DissimMatrixSecondHalf=dist(mdscale(DissimMatrixSecondHalf,5,'Criterion','metricstress','Start','random')');
    %%%% Correlation between first and second half %%%
    PickUpperTriangle=triu(ones(length(DissimMatrixFirstHalf),length(DissimMatrixFirstHalf)));
    PickUpperTriangle(eye(length(DissimMatrixFirstHalf))==1)=0;
    PickUpperTriangle=find(PickUpperTriangle);
    FirstHalf=DissimMatrixFirstHalf(PickUpperTriangle);
    SecondHalf=DissimMatrixSecondHalf(PickUpperTriangle);
    r=corr(FirstHalf,SecondHalf,'Type','Spearman');
    disp('********************')
    disp(['Subject: ' SubjIDs{n}]);
    disp(['first-second half corr (excluding missing cells):' num2str(r)]);
    

    imagesc(dissim_matrix_from_embedding,[0 1]);
    colormap(gray)
    axis off
    colorbar
    set(gca,'fontsize',16);
    print(gcf,['VisualizationDissimilarityMatrix-' SubjIDs{n} '.png'],'-dpng','-r300');
end
close all

%%%%%% Show in a 2D space %%%%%%%
for n=1:length(SubjIDs)

    Dissim{n}=Dissim{n}/max(Dissim{n}(:));
    D=mdscale(Dissim{n},2,'Criterion','metricstress');

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
    print(gcf,['2DVisualizationDissimilarityMatrix-' SubjIDs{n} '.png'],'-dpng','-r300');
end
close all;




function DissimMatrix=MakeDissimFromClickLoggings(ClickData,clicksListIndex,NMatrix)
    
    SimTracker=[];
    for i=1:NMatrix
        SimTracker{i}=[];
    end
    
    %%%%% Read Tragets and Clicks %%%%%
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
                    DissimMatrix(i,j)=1-0.5/(All+2);
                else
                    DissimMatrix(i,j)=1-NonOdds/(All+2);
                end
            end
        end
    end
end