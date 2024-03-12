clc;
clear;

SubjIDs={'960425785','674231075','664024137','596926644'};
Pairs=[3 5;13 20;4 12;14 27;15 24;14 16;5 17;10 17;7 12;1 28;13 30;22 23;2 10];

%%%%%%%% Data Address %%%%%%
CurrentAddress=pwd;
S=strsplit(CurrentAddress, filesep);
S(end)=[];
Data_JND_Address=[];
for i=1:length(S)
    Data_JND_Address=[Data_JND_Address S{i} filesep];
end
Data_JND_Address=[Data_JND_Address 'Collected Data' filesep 'Near Threshold Discrimination Task'];


for n=1:length(SubjIDs)
    
    %%%%% Load Staircases %%%%%%
    Files=dir(Data_JND_Address);
    for j=1:length(Files)
        Check=strsplit(Files(j).name,SubjIDs{n});
        if(length(Check)>1)
            break;
        end
    end
    Data_JND_Read=[Data_JND_Address filesep Files(j).name filesep];

    figure('Position', [0 0 1920 1080]);
    NDownslastTwenty=[];
    for i=1:size(Pairs,1)
        TextFile=fileread([Data_JND_Read 'staircase_' num2str(Pairs(i,1)) '-' num2str(Pairs(i,2)) '.log']);
        TextFile=strsplit(TextFile,'responseDistances');
        TextFile=strsplit(TextFile{2},'responseValues');

        DistanceVector=str2num(TextFile{1});

        %%%% number of downs in the last 20 trials %%%%%
        NDownslastTwenty(i)=sum(diff(DistanceVector(end-20:end))<0);

        TextFile=fileread([Data_JND_Read 'staircase_' num2str(Pairs(i,1)) '-' num2str(Pairs(i,2)) '.log']);
        TextFile=strsplit(TextFile,'responseValues');
        TextFile=strsplit(TextFile{2},'responseMorphIDs');
        TextFile=strsplit(TextFile{1},{'T','F'});

        AnswerVector=[];
        Count=1;
        for j=1:length(TextFile)
            if(length(TextFile{j})>3)
                if(strcmp(TextFile{j}(1:3),'rue'))
                    AnswerVector(Count)=1;
                    Count=Count+1;
                elseif(strcmp(TextFile{j}(1:4),'alse'))
                    AnswerVector(Count)=0;
                    Count=Count+1;
                end
            end
        end

        subplot(4,4,i)
        for j=1:length(DistanceVector)
            if(AnswerVector(j)==1)
                plot(j,DistanceVector(j),'o','Color',[0 0 1],'LineWidth',2);
            elseif(AnswerVector(j)==0)
                plot(j,DistanceVector(j),'o','Color',[1 0 0],'LineWidth',2);
            end
            hold on
        end
        plot(DistanceVector,'--','Color',[0 0 0]);
        grid on;
        title(['staircase-' num2str(Pairs(i,1)) '-' num2str(Pairs(i,2))]);
        ylabel('step')
        xlabel('trials')
        set(gca,'fontsize',12);
        ylim([0 1000]);

    end
    print(gcf,['VisualizationStaircase-' SubjIDs{n} '.png'],'-dpng','-r300');

    disp('****************************');
    disp(['SubjID: ' SubjIDs{n}])
    disp(['Number of staircases with less than 3 downs in their last 20 trials:' num2str(sum(NDownslastTwenty<3))])

end
close all;