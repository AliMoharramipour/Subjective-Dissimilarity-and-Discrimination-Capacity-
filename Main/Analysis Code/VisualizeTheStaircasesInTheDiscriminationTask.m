clc;
clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

JND_Pairs_In_Each_Sess=repmat({[3 4;3 13;4 7;5 30;6 14;7 28;9 14;11 22;14 30;15 19;19 20;20 27], [1 16;2 10;3 9;3 28;4 20;6 7;6 15;7 20;11 17;13 29;15 18;24 30]},1,11);
JND_Pairs_In_Each_Sess{end+1}=[11 22;19 20;24 30;3 28;1 16;4 7;7 28;6 15;13 29;2 10;3 9;5 30];
JND_Pairs_In_Each_Sess{end+1}=[15 18;3 4;6 7;4 20;3 13;14 30;15 19;6 14;7 20;9 14;11 17;20 27];

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

    disp('********************');
    disp(Files(j).name);

    figure('Position', [0 0 1920 1080]);
    NDownslastTwenty=[];
    for i=1:size(JND_Pairs_In_Each_Sess{n},1)
        TextFile=fileread([Data_JND_Read 'staircase_' num2str(JND_Pairs_In_Each_Sess{n}(i,1)) '-' num2str(JND_Pairs_In_Each_Sess{n}(i,2)) '.log']);
        TextFile=strsplit(TextFile,'responseDistances');
        TextFile=strsplit(TextFile{2},'responseValues');

        DistanceVector=str2num(TextFile{1});

        %%%% number of downs in the last 20 trials %%%%%
        NDownslastTwenty(i)=sum(diff(DistanceVector(end-20:end))<0);

        TextFile=fileread([Data_JND_Read 'staircase_' num2str(JND_Pairs_In_Each_Sess{n}(i,1)) '-' num2str(JND_Pairs_In_Each_Sess{n}(i,2)) '.log']);
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

        subplot(3,4,i)
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
        title(['staircase-' num2str(JND_Pairs_In_Each_Sess{n}(i,1)) '-' num2str(JND_Pairs_In_Each_Sess{n}(i,2))]);
        ylabel('step')
        xlabel('trials')
        set(gca,'fontsize',12);
        ylim([0 1000]);
        yticks([0 200 400 600 800 1000]);

    end
    print(gcf,['VisualizationStaircase-' SubjIDs{n} '.png'],'-dpng','-r300');
    
    disp(['SubjID: ' SubjIDs{n}]);
    disp(['Number of staircases with less than 3 downs in their last 20 trials:' num2str(sum(NDownslastTwenty<3))]);

end
close all;