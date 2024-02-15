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